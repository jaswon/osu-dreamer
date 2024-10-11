
from pathlib import Path

from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl

from .dataset import FullSequenceDataset, SubsequenceDataset
    

class Data(pl.LightningDataModule):
    def __init__(
        self,
        
        seq_len: int,
        subseq_density: float,
        batch_size: int,
        num_workers: int,
        
        data_path: str = "./data",
        val_split: float = 0.,
        val_size: int = 0,
    ):
        super().__init__()
        
        self.seq_len = seq_len
        
        self.subseq_density = subseq_density
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        if (val_split == 0) == (val_size == 0):
            raise ValueError('exactly one of `val_split` or `val_size` must be specified')
        self.val_split = val_split
        self.val_size = val_size
        
        # check if data dir exists
        self.data_dir = Path(data_path)
        if not self.data_dir.exists():
            raise ValueError(f'data dir `{self.data_dir}` does not exist, generate dataset first')
        
        # data dir exists, check for samples
        try:
            next(self.data_dir.rglob("*.map.pt"))
        except StopIteration:
            raise ValueError(f'data dir `{self.data_dir}` is empty, generate dataset first')
            
    def setup(self, stage: str):
        full_set = list(self.data_dir.rglob("*.map.pt"))
        
        val_size = self.val_size + int(len(full_set) * self.val_split)
            
        if val_size < 0:
            raise ValueError("`val_size` is negative")
            
        if val_size > len(full_set):
            raise RuntimeError(f"`val_size` ({val_size}) is greater than the number of samples ({len(full_set)})")
            
        train_size = len(full_set) - val_size
        print(f'train: {train_size} | val: {val_size}')
        train_split, val_split = random_split(full_set, [train_size, val_size]) # type: ignore
        
        self.train_set = SubsequenceDataset(
            dataset=train_split,
            seq_len=self.seq_len,
            subseq_density=self.subseq_density,
        )
        self.val_set = FullSequenceDataset(
            dataset=val_split,
            seq_len=self.seq_len,
        )

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
            batch_size=1,
            num_workers=self.num_workers,
        )