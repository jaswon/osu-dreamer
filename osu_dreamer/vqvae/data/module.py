
from pathlib import Path

from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl

from .dataset import FullSequenceDataset, SubsequenceDataset


class Data(pl.LightningDataModule):
    def __init__(
        self,
        
        seq_len: int,
        batch_size: int,
        num_workers: int,
        
        val_size: float | int,
        data_path: str = "./data",
    ):
        super().__init__()
        
        self.seq_len = seq_len
        
        self.batch_size = batch_size
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

            
    def setup(self, stage: str):
        
        train_size = len(self.full_set) - self.val_size
        print(f'train: {train_size} | val: {self.val_size}')
        train_split, val_split = random_split(self.full_set, [train_size, self.val_size])
        
        self.train_set = SubsequenceDataset(
            dataset=train_split,
            seq_len=self.seq_len,
        )
        self.val_set = FullSequenceDataset(
            dataset=val_split,
            seq_len=self.seq_len,
        )
            
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=1,
            num_workers=self.num_workers,
        )