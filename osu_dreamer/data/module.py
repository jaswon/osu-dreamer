
from typing import Iterator, NamedTuple
from jaxtyping import Float
from torch import Tensor

from pathlib import Path
import random

import torch as th
from torch.utils.data import random_split, Dataset, IterableDataset, DataLoader

import pytorch_lightning as pl

from .reclaim_memory import reclaim_memory
from .load_audio import A_DIM, read_spec
from .beatmap.encode import X_DIM, NUM_LABELS, read_beatmap


class Batch(NamedTuple):
    audio: Float[Tensor, str(f"{A_DIM} L")]
    chart: Float[Tensor, str(f"{X_DIM} L")]
    labels: Float[Tensor, str(f"{NUM_LABELS}")]


class BeatmapDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        num_workers: int,
        val_size: float | int,
        data_path: str = "./data",
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_workers = num_workers
        
        # check if data dir exists
        self.data_dir = Path(data_path)
        if not self.data_dir.exists():
            raise ValueError(f'data dir `{self.data_dir}` does not exist, generate dataset first')
        
        # data dir exists, check for samples
        self.full_set = list(self.data_dir.rglob("*.map.npy"))
        if len(self.full_set) == 0:
            raise ValueError(f'data dir `{self.data_dir}` is empty, generate dataset first')
        
        # check validation size
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
        
    def make_train_set(self, split) -> Dataset:
        return BatchedSignalDataset(self.seq_len, split)
    
    def make_val_set(self, split) -> Dataset:
        return SignalDataset(split)
            
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, num_workers=self.num_workers, pin_memory=True, persistent_workers=True)
            
    def setup(self, stage: str):
        train_size = len(self.full_set) - self.val_size
        print(f'train: {train_size} | val: {self.val_size}')
        train_split, val_split = random_split(
            self.full_set, # type: ignore
            [train_size, self.val_size],
        )
        
        self.train_set = self.make_train_set(train_split)
        self.val_set = self.make_val_set(val_split)
    

class SignalDataset(IterableDataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        
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
                yield from self.make_samples(map_file, i)
            finally:
                reclaim_memory()

    def make_samples(self, map_file: Path, map_idx: int) -> Iterator[Batch]:
        with open(map_file.parent / "spec.npy", "rb") as f:
            audio = th.from_numpy(read_spec(f)).float()
        with open(map_file, 'rb') as f:
            chart_arr, label_arr = read_beatmap(f)
            chart = th.from_numpy(chart_arr).float()
            labels = th.from_numpy(label_arr).float()
        yield Batch(audio,chart,labels)


class BatchedSignalDataset(SignalDataset):
    def __init__(self, seq_len: int, dataset):
        super().__init__(dataset)
        self.seq_len = seq_len

    def make_samples(self, map_file: Path, map_idx: int) -> Iterator[Batch]:
        audio,chart,labels = next(super().make_samples(map_file, map_idx))
        offset_end = chart.size(-1)-self.seq_len+1
        if offset_end < 1:
            return
        offset_start = th.randint(0, min(self.seq_len, offset_end), ()).item()
        for i in th.arange(offset_start, offset_end, self.seq_len):
            yield Batch(audio[...,i:i+self.seq_len], chart[...,i:i+self.seq_len],labels)