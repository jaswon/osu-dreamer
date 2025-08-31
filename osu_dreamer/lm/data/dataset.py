from typing import NamedTuple, Iterator
from torch import Tensor
from jaxtyping import Float

import pickle
import random
from pathlib import Path
from collections.abc import Iterator

import numpy as np

import torch as th
from torch.utils.data import IterableDataset

from osu_dreamer.data.reclaim_memory import reclaim_memory
from osu_dreamer.lm.data.tokens.tokenizer import Tokenizer
from osu_dreamer.lm.data.parse.parse_beatmap import from_beatmap


class Batch(NamedTuple):
    audio: Float[Tensor, "B T"]  # Full audio per sample
    map_features: Float[Tensor, "B M"]
    tokens: Float[Tensor, "B L"]  # Subsampled token sequences
    timestamps: Float[Tensor, "B L"]  # Token timestamps in ms
    audio_lengths: Float[Tensor, "B"]  # Audio lengths for padding


class Dataset(IterableDataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.dataset = kwargs.pop("dataset")
        self.seq_len: int = kwargs.pop("seq_len")
        self.tokenizer: Tokenizer = kwargs.pop("tokenizer")
            
        if len(kwargs):
            raise ValueError(f"unexpected kwargs: {kwargs}")
        
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
        
        dataset = sorted(self.dataset)
        for i, map_file in random.sample(list(enumerate(dataset)), int(len(dataset))):
            if i % num_workers != worker_id:
                continue
                
            try:
                yield from self.sample_map(map_file, i)
            finally:
                reclaim_memory()
            
    def sample_map(self, map_file: Path, map_idx: int) -> Iterator[Batch]:
        audio = th.tensor(np.load(map_file.parent / "spec.pt")).float() # A L
        audio_length = audio.size(1)
        
        # Load and tokenize beatmap
        with open(map_file, 'rb') as f:
            bm = pickle.load(f)
        
        ibm, diff, _ = from_beatmap(bm)
        beatmap_tokens, token_timestamps = self.tokenizer.encode(ibm)
        
        # Subsample tokens if sequence is too long
        if len(beatmap_tokens) > self.seq_len:
            # Random start position for subsampling
            start_idx = random.randint(0, len(beatmap_tokens) - self.seq_len)
            end_idx = start_idx + self.seq_len
            
            # Subsample tokens and corresponding timestamps
            subsampled_tokens = beatmap_tokens[start_idx:end_idx]
            subsampled_timestamps = token_timestamps[start_idx:end_idx]
        else:
            # Pad if sequence is too short
            subsampled_tokens = beatmap_tokens + [0] * (self.seq_len - len(beatmap_tokens))  # 0 = PAD token
            subsampled_timestamps = token_timestamps + [0] * (self.seq_len - len(token_timestamps))
        
        # Convert to tensors
        audio_tensor = th.tensor(audio).float()
        map_features = th.tensor([
            diff.hp_drain_rate,
            diff.circle_size,
            diff.overall_difficulty,
            diff.approach_rate,
            diff.slider_tick_rate,
        ]).float()
        tokens_tensor = th.tensor(subsampled_tokens).long()
        timestamps_tensor = th.tensor(subsampled_timestamps).float()
        audio_length_tensor = th.tensor([audio_length]).long()
        
        yield Batch(audio_tensor, map_features, tokens_tensor, timestamps_tensor, audio_length_tensor)


class DataModule:
    def __init__(
        self,
        seq_len: int,
        batch_size: int,
        num_workers: int,
        val_size: float | int,
        data_path: str = "./data",
        vocab_config = None,  # VocabConfig instance
    ):
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.data_path = Path(data_path)
        
        # Initialize tokenizer
        if vocab_config is None:
            from osu_dreamer.lm.data.tokens.tokens import VocabConfig
            vocab_config = VocabConfig()
        self.tokenizer = Tokenizer(vocab_config)
        
        # Check if data dir exists
        if not self.data_path.exists():
            raise ValueError(f'data dir `{self.data_path}` does not exist, generate dataset first')
        
        # Get all map files
        self.full_set = list(self.data_path.rglob("*.map.pkl"))
        if len(self.full_set) == 0:
            raise ValueError(f'data dir `{self.data_path}` is empty, generate dataset first')
        
        # Validate val_size
        if val_size <= 0:
            raise ValueError(f'invalid {val_size=}')
        elif val_size < 1:
            # interpret as fraction of full set
            val_size = int(len(self.full_set) * val_size)
            if val_size == 0:
                raise ValueError(f'empty validation set, given {val_size=} and {len(self.full_set)=}')
        else:
            # interpret as number of samples
            val_size = round(val_size)
            if val_size > len(self.full_set):
                raise ValueError(f"{val_size=} is greater than {len(self.full_set)=}")
        self.val_size = val_size
        
        # Split dataset
        train_size = len(self.full_set) - self.val_size
        print(f'train: {train_size} | val: {self.val_size}')
        
        # Shuffle and split
        random.shuffle(self.full_set)
        self.train_set = self.full_set[:train_size]
        self.val_set = self.full_set[train_size:]
        
        # Create datasets
        self.train_dataset = Dataset(
            dataset=self.train_set,
            seq_len=self.seq_len,
            tokenizer=self.tokenizer,
        )
        self.val_dataset = Dataset(
            dataset=self.val_set,
            seq_len=self.seq_len,
            tokenizer=self.tokenizer,
        )
    
    def train_dataloader(self):
        return th.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
    
    def val_dataloader(self):
        return th.utils.data.DataLoader(
            self.val_dataset,
            batch_size=1,  # Validation with batch size 1 for simplicity
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
    
    def collate_fn(self, batch):
        """Custom collate function to handle variable audio lengths"""
        audios, map_features, tokens, timestamps, audio_lengths = zip(*batch)
        
        # Pad audio to max length in batch
        max_audio_len = max(len(audio) for audio in audios)
        padded_audios = []
        for audio in audios:
            if len(audio) < max_audio_len:
                padded_audio = th.cat([audio, th.zeros(max_audio_len - len(audio))])
            else:
                padded_audio = audio
            padded_audios.append(padded_audio)
        
        # Stack tensors
        audio_batch = th.stack(padded_audios)
        map_feature_batch = th.stack(map_features)
        tokens_batch = th.stack(tokens)
        timestamps_batch = th.stack(timestamps)
        audio_lengths_batch = th.stack(audio_lengths)
        
        return Batch(audio_batch, map_feature_batch, tokens_batch, timestamps_batch, audio_lengths_batch) 