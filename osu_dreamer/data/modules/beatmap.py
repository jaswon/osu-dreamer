
from typing import Iterator, NamedTuple
from jaxtyping import Float
from torch import Tensor

from pathlib import Path
import random

import torch as th
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

import pytorch_lightning as pl

from ..reclaim_memory import reclaim_memory
from ..load_audio import A_DIM, read_spec
from ..beatmap.encode import X_DIM, NUM_LABELS, BeatmapEncoding, read_beatmap


class Batch(NamedTuple):
    audio: Float[Tensor, str(f"{A_DIM} L")]
    chart: Float[Tensor, str(f"{X_DIM} L")]
    labels: Float[Tensor, str(f"{NUM_LABELS}")]


def pad_to_multiple(x: Float[Tensor, "... L"], chunk_size: int) -> Float[Tensor, "... Lp"]:
    """right-pad the time axis so its length is a multiple of `chunk_size` —
    the models' encoders require chunk-aligned inputs."""
    pad = (chunk_size - x.size(-1) % chunk_size) % chunk_size
    return F.pad(x, (0, pad), mode='replicate') if pad > 0 else x


def hold_out_mapsets(data_dir: Path, pattern: str, max_val_count: int, max_val_frac: float) -> set[str]:
    """hold out whole mapsets (all difficulties of a song) to prevent
    train/val leakage via shared audio. `max_val_size` is a map count: mapsets
    are drawn until at most `max_val_size` maps are held out."""

    if not data_dir.exists():
        raise ValueError(f'data dir `{data_dir}` does not exist, generate dataset first')
    
    # data dir exists, check for samples
    full_size = sum(1 for _ in data_dir.rglob(pattern))
    if full_size == 0:
        raise ValueError(f'data dir `{data_dir}` is empty, generate dataset first')
    
    # check validation size
    if max_val_count <= 0:
        raise ValueError(f'invalid {max_val_count=}')
    
    if not (0 < max_val_frac < 1):
        raise ValueError(f'invalid {max_val_frac=}')
    
    max_val_size = min(max_val_count, int(full_size * max_val_frac))
    if not (0 < max_val_size < full_size):
        raise ValueError(f'invalid {max_val_size=} given {full_size=} {max_val_count=} {max_val_frac=}')
    
    mapsets = set()
    val_size = 0
    for mapset in data_dir.iterdir():
        mapset_count = sum(1 for _ in mapset.glob(pattern))
        if val_size + mapset_count > max_val_size:
            break
        val_size += mapset_count
        mapsets.add(mapset.stem)

    print(f'train: {full_size - val_size} | val: {val_size}')
    return mapsets


class BeatmapDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        num_workers: int,
        max_val_count: int = 512,
        max_val_frac: float = .3,
        data_path: str = "./data",
        shuffle_buffer_size: int = 1,
        max_per_map: int = -1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_per_map = max_per_map
        
        # check if data dir exists
        data_dir = Path(data_path)
        val_sets = hold_out_mapsets(data_dir, '*.map.npy', max_val_count, max_val_frac)
        self.train_set = BatchedSignalDataset(data_dir, dict(exclude=val_sets), self.seq_len, self.shuffle_buffer_size, self.max_per_map)
        self.val_set = SignalDataset(data_dir, dict(include=val_sets))
            
    def train_dataloader(self):
        return DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=True, 
            persistent_workers=True, 
            drop_last=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_set, 
            batch_size=1, 
            num_workers=self.num_workers, 
            pin_memory=True, 
            persistent_workers=True,
        )
            

class SignalDataset(IterableDataset):
    def __init__(self, data_dir: Path, mapsets: dict[str, set[str]], shuffle_buffer_size: int = 1):
        super().__init__()
        if 'include' in mapsets:
            filter_fn = lambda sample: sample.parent.name in mapsets['include']
        elif 'exclude' in mapsets:
            filter_fn = lambda sample: sample.parent.name not in mapsets['exclude']
        else:
            raise ValueError('neither include nor exclude provided')
        self.dataset = list(filter(filter_fn, data_dir.rglob("*.latent.npz")))
        self.shuffle_buffer_size = shuffle_buffer_size

    def _sample_stream(self, num_workers: int, worker_id: int) -> Iterator[Batch]:
        for i, map_file in enumerate(self.dataset):
            if i % num_workers != worker_id:
                continue
            try:
                yield from self.make_samples(map_file, i)
            finally:
                reclaim_memory()

    def __iter__(self):
        worker_info = th.utils.data.get_worker_info() # type: ignore
        if worker_info is None:  
            # single-process data loading, return the full iterator
            num_workers = 1
            worker_id = 0
            seed = th.initial_seed()
        else:  # in a worker process
            # split workload
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            seed = worker_info.seed
        
        random.seed(seed)
        
        stream = self._sample_stream(num_workers, worker_id)
        if self.shuffle_buffer_size <= 1:
            yield from stream
            return

        # shuffle buffer
        buffer: list[Batch] = []
        for sample in stream:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(sample)
                continue
            j = random.randrange(len(buffer))
            yield buffer[j]
            buffer[j] = sample
        random.shuffle(buffer)
        yield from buffer

    def make_samples(self, map_file: Path, map_idx: int) -> Iterator[Batch]:
        with open(map_file.parent / "spec.npy", "rb") as f:
            audio = th.from_numpy(read_spec(f)).float()
        with open(map_file, 'rb') as f:
            chart_arr, label_arr = read_beatmap(f)
            chart = th.from_numpy(chart_arr).float()
            labels = th.from_numpy(label_arr).float()
        yield Batch(audio,chart,labels)


class BatchedSignalDataset(SignalDataset):
    def __init__(
        self, 
        data_dir: Path, 
        mapsets: dict[str, set[str]], 
        seq_len: int, 
        shuffle_buffer_size: int = 1,
        max_per_map: int = -1, 
    ):
        super().__init__(data_dir, mapsets, shuffle_buffer_size)
        self.seq_len = seq_len
        self.max_per_map = max_per_map if max_per_map > 0 else float('inf')

    def make_samples(self, map_file: Path, map_idx: int) -> Iterator[Batch]:
        audio,chart,labels = next(super().make_samples(map_file, map_idx))
        offset_end = chart.size(-1)-self.seq_len+1
        if offset_end < 1:
            return
        offset_start = th.randint(0, min(self.seq_len, offset_end), ()).item()
        idxs = th.arange(offset_start, offset_end, self.seq_len)
        idxs = idxs[th.randperm(len(idxs))[:min(self.max_per_map, len(idxs))]]
        for i in idxs:
            chart_window = chart[...,i:i+self.seq_len].clone()
            
            # flip augment
            if th.rand(()) < 0.5:
                chart_window[BeatmapEncoding.X].mul_(-1).add_(1)
            if th.rand(()) < 0.5:
                chart_window[BeatmapEncoding.Y].mul_(-1).add_(1)

            # clone the audio window so the buffer doesn't pin the full spec
            yield Batch(audio[...,i:i+self.seq_len].clone(), chart_window, labels)