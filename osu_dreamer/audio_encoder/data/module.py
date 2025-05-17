
from typing import NamedTuple
from torch import Tensor
from jaxtyping import Float, Int

import pickle
from pathlib import Path
from collections.abc import Iterator

import numpy as np

import torch as th
from torch.utils.data import DataLoader

from osu_dreamer.data.module import BeatmapDataModule, BeatmapDataset
from osu_dreamer.data.load_audio import A_DIM, get_frame_times

from .events import beatmap_events


class Batch(NamedTuple):
    audio: Float[Tensor, str(f"{A_DIM} L")]
    events: Int[Tensor, "L"]
    

class BeatmapEventDataset(BeatmapDataset):
    def __init__(self, dataset, seq_len: int):
        super().__init__(dataset)
        self.seq_len = seq_len

    def make_samples(self, map_file: Path, map_idx: int) -> Iterator[Batch]:
        audio = th.tensor(np.load(map_file.parent / "spec.pt")).float() # A L
        frame_times = get_frame_times(audio.size(-1))
        with open(map_file, 'rb') as f:
            bm = pickle.load(f)
        events = th.tensor(beatmap_events(bm, frame_times)).long()
            
        yield Batch(audio,events)

        
class SubsequenceBeatmapEventDataset(BeatmapEventDataset):
    def make_samples(self, map_file: Path, map_idx: int):
        for audio,events in super().make_samples(map_file, map_idx):
            L = audio.size(-1)
            if self.seq_len >= L:
                return

            num_samples = int(L / self.seq_len)
            for idx in th.randperm(L - self.seq_len)[:num_samples]:
                sl = ..., slice(idx,idx+self.seq_len)
                yield Batch(audio[sl], events[sl])


class Data(BeatmapDataModule):
    def __init__(
        self,
        
        seq_len: int,
        batch_size: int,

        num_workers: int,
        val_size: float | int,
        data_path: str = "./data",
    ):
        super().__init__(num_workers, val_size, data_path)
        self.seq_len = seq_len
        self.batch_size = batch_size
            
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
    
    def make_train_set(self, split) -> SubsequenceBeatmapEventDataset:
        return SubsequenceBeatmapEventDataset(split, self.seq_len)
    
    def make_val_set(self, split) -> BeatmapEventDataset:
        return BeatmapEventDataset(split, self.seq_len)
            
