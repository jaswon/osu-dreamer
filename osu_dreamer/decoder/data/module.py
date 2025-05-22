
from typing import NamedTuple
from torch import Tensor
from jaxtyping import Float, Int

import pickle
from pathlib import Path
from collections.abc import Iterator

import numpy as np

import torch as th
from torch.utils.data import Dataset, DataLoader

from osu_dreamer.data.module import BeatmapDataModule, BeatmapDataset
from osu_dreamer.data.load_audio import A_DIM
from osu_dreamer.data.labels import NUM_LABELS, get_labels

from .tokenize import tokenize


class Batch(NamedTuple):
    labels: Float[Tensor, str(f"B {NUM_LABELS}")]
    audio: Float[Tensor, str(f"B {A_DIM} L")]
    types: Int[Tensor, "B N"]
    tokens: Int[Tensor, "B N"]
    timestamps: Float[Tensor, "B N"]
    positions: Float[Tensor, "B N 2"]

class TokenDataset(BeatmapDataset):
    def __init__(self, max_audio_len: int, dataset):
        super().__init__(dataset)
        self.max_audio_len = max_audio_len

    def make_samples(self, map_file: Path, map_idx: int) -> Iterator[Batch]:
        audio = np.load(map_file.parent / "spec.pt")
        if audio.shape[1] > self.max_audio_len:
            return
        with open(map_file, 'rb') as f:
            bm = pickle.load(f)
        types, tokens, timestamps, positions = tokenize(bm)
        labels = get_labels(bm)
        
        yield Batch(
            th.tensor(labels).float(),
            th.tensor(audio).float(),
            th.tensor(types).long(),
            th.tensor(tokens).long(),
            th.tensor(timestamps).float(),
            th.tensor(positions).float(),
        )

    
class Data(BeatmapDataModule):
    def __init__(
        self, 
        max_audio_len: int,
        num_workers: int, 
        val_size: float | int, 
        data_path: str = "./data",
    ):
        super().__init__(num_workers, val_size, data_path)
        self.max_audio_len = max_audio_len
        
    def make_train_set(self, split) -> Dataset:
        return TokenDataset(self.max_audio_len, split)
    
    def make_val_set(self, split) -> Dataset:
        return TokenDataset(self.max_audio_len, split)
            
    def train_dataloader(self):
        return DataLoader(self.train_set, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, num_workers=self.num_workers)