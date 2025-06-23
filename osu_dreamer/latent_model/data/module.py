
from typing import NamedTuple
from torch import Tensor
from jaxtyping import Float

import pickle
from pathlib import Path
from collections.abc import Iterator

import numpy as np

import torch as th
from torch.utils.data import Dataset, DataLoader

from osu_dreamer.data.module import BeatmapDataModule, BeatmapDataset
from osu_dreamer.data.load_audio import A_DIM, get_frame_times
from osu_dreamer.data.beatmap.encode import X_DIM, encode_beatmap

class Batch(NamedTuple):
    audio: Float[Tensor, str(f"B {A_DIM} L")]
    chart: Float[Tensor, str(f"B {X_DIM} L")]

class SignalDataset(BeatmapDataset):
    def make_samples(self, map_file: Path, map_idx: int) -> Iterator[Batch]:
        audio = th.tensor(np.load(map_file.parent / "spec.pt")).float()
        L = audio.size(1)
        with open(map_file, 'rb') as f:
            bm = pickle.load(f)
        chart = th.tensor(encode_beatmap(bm, get_frame_times(L))).float()
        yield Batch(audio,chart)

class BatchedSignalDataset(SignalDataset):
    def __init__(self, seq_len: int, dataset):
        super().__init__(dataset)
        self.seq_len = seq_len

    def make_samples(self, map_file: Path, map_idx: int) -> Iterator[Batch]:
        audio,chart = next(super().make_samples(map_file, map_idx))
        L = chart.size(-1)
        for i in th.arange(0, L-self.seq_len+1, self.seq_len):
            yield Batch(audio[...,i:i+self.seq_len], chart[...,i:i+self.seq_len])
    
class Data(BeatmapDataModule):
    def __init__(
        self, 
        batch_size: int,
        seq_len: int,
        num_workers: int, 
        val_size: float | int, 
        data_path: str = "./data",
    ):
        super().__init__(num_workers, val_size, data_path)
        self.batch_size = batch_size
        self.seq_len = seq_len
        
    def make_train_set(self, split) -> Dataset:
        return BatchedSignalDataset(self.seq_len, split)
    
    def make_val_set(self, split) -> Dataset:
        return SignalDataset(split)
            
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=1, num_workers=self.num_workers)