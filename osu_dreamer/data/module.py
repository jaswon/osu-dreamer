
from typing import Iterator

from pathlib import Path
import random

import torch as th
from torch.utils.data import random_split, Dataset, IterableDataset

import pytorch_lightning as pl

from osu_dreamer.data.reclaim_memory import reclaim_memory


class BeatmapDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_workers: int,
        val_size: float | int,
        data_path: str = "./data",
    ):
        super().__init__()
        self.num_workers = num_workers
        
        # check if data dir exists
        self.data_dir = Path(data_path)
        if not self.data_dir.exists():
            raise ValueError(f'data dir `{self.data_dir}` does not exist, generate dataset first')
        
        # data dir exists, check for samples
        self.full_set = list(self.data_dir.rglob("*.map.pkl"))
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
        raise NotImplementedError()
    
    def make_val_set(self, split) -> Dataset:
        raise NotImplementedError()
            
    def setup(self, stage: str):
        train_size = len(self.full_set) - self.val_size
        print(f'train: {train_size} | val: {self.val_size}')
        train_split, val_split = random_split(
            self.full_set, # type: ignore
            [train_size, self.val_size],
        )
        
        self.train_set = self.make_train_set(train_split)
        self.val_set = self.make_val_set(val_split)
    

class BeatmapDataset(IterableDataset):
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

    def make_samples(self, map_file: Path, map_idx: int) -> Iterator:
        raise NotImplementedError()