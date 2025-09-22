from typing import NamedTuple, Iterator
from torch import Tensor
from jaxtyping import Float, Int

import pickle
import random
from pathlib import Path
from collections.abc import Iterator

import numpy as np

import torch as th
from torch.utils.data import IterableDataset

import pytorch_lightning as pl

from osu_dreamer.data.reclaim_memory import reclaim_memory
from osu_dreamer.lm.data.tokens.tokenizer import Tokenizer
from osu_dreamer.lm.data.tokens.tokens import Token, TokenType, Vocab


class Batch(NamedTuple):
    map_features: Float[Tensor, "B M"]  # Map features
    audio: Float[Tensor, "B A L"]       # Spectrogram
    seq: Int[Tensor, "B N+1"]           # token sequences


class Dataset(IterableDataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.dataset = kwargs.pop("dataset")
        self.context_size: int = kwargs.pop("context_size")
        self.vocab: Vocab = kwargs.pop("vocab")
            
        if len(kwargs):
            raise ValueError(f"unexpected kwargs: {kwargs}")
        
    def __iter__(self) -> Iterator[Batch]:
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
        for i, map_file in random.sample(list(enumerate(dataset)), len(dataset)):
            if i % num_workers != worker_id:
                continue
                
            try:
                yield from self.sample_map(map_file, i)
            finally:
                reclaim_memory()
            
    def sample_map(self, map_file: Path, map_idx: int) -> Iterator[Batch]:

        audio = th.from_numpy(np.load(map_file.parent / "spec.pt")).float()

        if audio.size(-1) < self.vocab.time_bins:
            return

        with open(map_file, 'rb') as f:
            ibm, diff = pickle.load(f)
        
        map_features = th.tensor([
            diff.hp_drain_rate,
            diff.circle_size,
            diff.overall_difficulty,
            diff.approach_rate,
            diff.slider_tick_rate,
        ]).float()

        try:
            tokenizer = Tokenizer(self.vocab, ibm)
        except Exception as e:
            raise Exception(map_file) from e

        pad = self.vocab.ids[Token(TokenType.PAD)]
        num_starts = audio.size(-1) - self.vocab.time_bins + 1
        for start_idx in th.randperm(num_starts)[:1 + num_starts // self.vocab.time_bins]:

            seq = tokenizer.encode(int(start_idx.item()))
            if len(seq) < self.context_size:
                # pad
                num_pad = self.context_size - len(seq)
                seq.extend([(pad, 256, 192, self.vocab.time_bins)] * num_pad)
            elif len(seq) > self.context_size:
                # random slice
                i = random.randrange(len(seq) - self.context_size + 1)
                seq = seq[i:i+self.context_size]

            yield Batch(
                map_features = map_features, 
                audio = audio[:, start_idx:start_idx + self.vocab.time_bins],
                seq = th.tensor(seq).long(),
            )


class Data(pl.LightningDataModule):
    def __init__(
        self,
        vocab: Vocab,
        context_size: int,
        batch_size: int,
        num_workers: int,
        val_size: float | int,
        data_path: str = "./data",
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.vocab = vocab
        self.context_size = context_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_size = val_size
        self.data_path = Path(data_path)
        
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
            context_size=self.context_size,
            vocab=self.vocab,
        )
        self.val_dataset = Dataset(
            dataset=self.val_set,
            context_size=self.context_size,
            vocab=self.vocab,
        )
    
    def train_dataloader(self):
        return th.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        return th.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )