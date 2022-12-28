import os
import random
import time
import warnings

import pickle

from pathlib import Path
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm

import numpy as np
import librosa


import torch
from torch.utils.data import IterableDataset, DataLoader, random_split

import pytorch_lightning as pl

from osu_dreamer.osu.beatmap import Beatmap
from osu_dreamer.tokens import TO_IDX, EOS, BSS, from_beatmap as tokens_from_beatmap

# audio processing constants
N_FFT = 2048
SR = 22000
HOP_LEN = (SR // 1000) * 4 # 4 ms per frame
N_MELS = 64

A_DIM = 40

# check if using WSL
if os.system("uname -r | grep microsoft > /dev/null") == 0:
    def reclaim_memory():
        """
        free the vm page cache - see `https://devblogs.microsoft.com/commandline/memory-reclaim-in-the-windows-subsystem-for-linux-2/`
        
        add to /etc/sudoers:
        %sudo ALL=(ALL) NOPASSWD: /bin/tee /proc/sys/vm/drop_caches
        """
        os.system("echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null")
else:
    def reclaim_memory():
        pass

def load_audio(audio_file):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wave, _ = librosa.load(audio_file, sr=SR, res_type='polyphase', mono=True)

    # ensure first and last frames are zeroed
    wave[:HOP_LEN] = 0
    wave[-HOP_LEN:] = 0

    # compute spectrogram
    return librosa.feature.mfcc(
        y=wave,
        sr=SR,
        n_mfcc=A_DIM,
        n_fft=N_FFT,
        hop_length=HOP_LEN,
        n_mels=N_MELS,
    )

def prepare_map(data_dir, map_file):
    try:
        bm = Beatmap(map_file)
    except Exception as e:
        print(f"{map_file}: {e}")
        return

    af_dir = "_".join([bm.audio_filename.stem, *(s[1:] for s in bm.audio_filename.suffixes)])
    map_dir = data_dir / map_file.parent.name / af_dir
    
    spec_path =  map_dir / "spec.npy"

    if not spec_path.exists():
        # load audio file
        try:
            spec = load_audio(bm.audio_filename)
        except Exception as e:
            print(f"{bm.audio_filename}: {e}")
            return

        # save spectrogram
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(spec_path, spec, allow_pickle=False)

    # compute tokens
    with open(map_dir / f"{map_file.stem}.map.pickle", "wb") as f:
        pickle.dump(tokens_from_beatmap(bm), f)
    

class Data(pl.LightningDataModule):
    def __init__(
        self,
        
        seq_depth: int,
        sample_density: float,
        subseq_density: float,
        context_len: int,
        batch_size: int,
        num_workers: int,
        
        data_path: str = "./data",
        src_path: str = None,
        val_split: float = None,
        val_size: int = None,
    ):
        super().__init__()
        
        self.seq_len = 2 ** seq_depth
        
        self.sample_density = sample_density
        self.subseq_density = subseq_density
        self.context_len = context_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        if (val_split is None) == (val_size is None):
            raise ValueError('exactly one of `val_split` or `val_size` must be specified')
        self.val_split = val_split
        self.val_size = val_size
        
        self.data_dir = Path(data_path)
        try:
            self.data_dir.mkdir()
        except FileExistsError:
            # data dir exists, check for samples
            try:
                next(self.data_dir.rglob("*.map.pickle"))
                # data dir already has samples
                self.src_dir = None
                return
            except StopIteration:
                # data dir is empty, must prepare data
                pass
            
        if src_path is None:
            raise ValueError("`src_path` must be specified when `data_dir` does not exist or is empty")
        self.src_dir = Path(src_path)
        
        
    def prepare_data(self):
        if self.src_dir is None:
            return
        
        src_maps = list(self.src_dir.rglob("*.osu"))
        print(f"{len(src_maps)} osu! beatmaps found, processing...")
        with Pool(processes=self.num_workers) as p:
            for _ in tqdm(p.imap_unordered(partial(prepare_map, self.data_dir), src_maps), total=len(src_maps)):
                reclaim_memory()
            
    def setup(self, stage: str):
        full_set = list(self.data_dir.rglob("*.map.pickle"))
        
        if self.val_size is not None:
            val_size = self.val_size
        else:
            val_size = int(len(full_set) * self.val_split)
            
        if val_size < 0:
            raise ValueError("`val_size` is negative")
            
        if val_size > len(full_set):
            raise ValueError(f"`val_size` ({val_size}) is greater than the number of samples ({len(full_set)})")
            
        train_size = len(full_set) - val_size
        print(f'train: {train_size} | val: {val_size}')
        train_split, val_split = random_split(full_set, [train_size, val_size])
        
        dataset_kwargs = dict(
            seq_len=self.seq_len,
            sample_density=self.sample_density,
            subseq_density=self.subseq_density,
            context_len=self.context_len,
        )
        self.train_set = Dataset(dataset=train_split, **dataset_kwargs)
        self.val_set = Dataset(dataset=val_split, **dataset_kwargs)

        print('approximate epoch length:', self.train_set.approx_dataset_size / self.batch_size)
            
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    
class Dataset(IterableDataset):
    def __init__(self, *, dataset, seq_len, subseq_density, sample_density, context_len, **kwargs):
        super().__init__()
        self.dataset = dataset
        self.sample_density = sample_density
        self.seq_len = seq_len
        self.subseq_density = subseq_density
        self.context_len = context_len
        
        if not 0 < self.sample_density <= 1:
            raise ValueError("sample density must be in (0, 1]:", self.sample_density)
            
        if len(kwargs):
            raise ValueError(f"unexpected kwargs: {kwargs}")

        num_samples = 0
        for map_file in self.dataset:
            with open(map_file.parent / "spec.npy", 'rb') as f:
                magic = np.lib.format.read_magic(f)
                read_header = np.lib.format.read_array_header_1_0 if magic[0] == 1 else np.lib.format.read_array_header_2_0
                shape = read_header(f, max_header_size=100000)[0]
                num_samples += int(shape[-1] / self.seq_len * self.subseq_density)
        
        self.approx_dataset_size = num_samples * self.sample_density
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  
            # single-process data loading, return the full iterator
            num_workers = 1
            worker_id = 0
            seed = torch.initial_seed()
        else:  # in a worker process
            # split workload
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            seed = worker_info.seed
        
        random.seed(seed)
        
        dataset = sorted(self.dataset)
        for i, sample in random.sample(list(enumerate(dataset)), int(len(dataset) * self.sample_density)):
            if i % num_workers != worker_id:
                continue
                
            try:
                for x in self.samples_for_map(sample):
                    yield x
            finally:
                reclaim_memory()

    def samples_for_map(self, map_file):
        a: "A,L" = torch.tensor(np.load(map_file.parent / "spec.npy")).float()
        
        L = a.size(-1)
        if self.seq_len >= L:
            return

        num_samples = int(L / self.seq_len * self.subseq_density)

        with open(map_file, 'rb') as f:
            sentences, sentence_starts, sentence_ends = pickle.load(f)
        sentence_starts = np.array(sentence_starts)
        sentence_ends = np.array(sentence_ends)

        def tokens_for_range(start, end):
            """
            returns all tokens between `start` and `end` inclusive

            - toks: 1D array of ints that are keys into `TO_IDX`
            - times: 1D array of ints that are frame indices (that are relative to `start`) for `TIME`-type tokens
            """
            toks = [BSS] * self.context_len
            times = [-1] * self.context_len
            for idx in np.nonzero((start <= sentence_starts) & (end >= sentence_ends))[0]:
                for tok in sentences[idx]:
                    if isinstance(tok, int):
                        toks.append(TO_IDX["TIME"])
                        times.append(tok - start)
                    else:
                        toks.append(TO_IDX[tok])
                        times.append(-1)

            toks.append(EOS)
            times.append(-1)

            return np.array(toks, dtype=int), librosa.time_to_frames(np.array(times), sr=SR, hop_length=HOP_LEN).round().astype(int)

        for idx in torch.randperm(L - self.seq_len)[:num_samples]:
            toks, times = tokens_for_range(*librosa.frames_to_time(np.array([idx, idx+self.seq_len]), sr=SR, hop_length=HOP_LEN))
            if len(toks) <= self.context_len:
                pad_amt = self.context_len - len(toks) + 1
                toks = np.pad(toks, (0, pad_amt), constant_values=EOS)
                times = np.pad(toks, (0, pad_amt), constant_values=-1)

            seen_context = torch.randperm(len(toks)-self.context_len)[0]
            yield tuple([
                a[:, idx:idx+self.seq_len].clone(),
                toks[seen_context:seen_context+self.context_len],
                times[seen_context:seen_context+self.context_len],
                toks[seen_context+1:seen_context+self.context_len+1],
                times[seen_context+1:seen_context+self.context_len+1],
            ])