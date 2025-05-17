
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

from .tokenize import to_tokens_and_timestamps


class Batch(NamedTuple):
    audio: Float[Tensor, str(f"{A_DIM} L")]
    tokens: Int[Tensor, "B N"]
    timestamps: Int[Tensor, "B N"]

class TokenDataset(BeatmapDataset):
    def make_samples(self, map_file: Path, map_idx: int) -> Iterator[Batch]:
        audio = np.load(map_file.parent / "spec.pt")
        with open(map_file, 'rb') as f:
            bm = pickle.load(f)
        tokens, timestamps = to_tokens_and_timestamps(bm)

        yield Batch(
            th.tensor(audio).float(),
            th.tensor(tokens).long(),
            th.tensor(timestamps).long(),
        )

    
class Data(BeatmapDataModule):
    def make_train_set(self, split) -> Dataset:
        return TokenDataset(split)
    
    def make_val_set(self, split) -> Dataset:
        return TokenDataset(split)
            
    def train_dataloader(self):
        return DataLoader(self.train_set, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, num_workers=self.num_workers)