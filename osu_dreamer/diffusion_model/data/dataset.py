
from typing import NamedTuple
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
from osu_dreamer.data.load_audio import A_DIM, get_frame_times
from osu_dreamer.data.beatmap.encode import X_DIM, encode_beatmap
from osu_dreamer.data.labels import NUM_LABELS, get_labels


class Batch(NamedTuple):
    audio: Float[Tensor, str(f"{A_DIM} L")]
    chart: Float[Tensor, str(f"{X_DIM} L")]
    labels: Float[Tensor, str(f"{NUM_LABELS}")]


class FullSequenceDataset(IterableDataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.dataset = kwargs.pop("dataset")
        self.seq_len = kwargs.pop("seq_len")
            
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
        with open(map_file, 'rb') as f:
            bm = pickle.load(f)
        chart = th.tensor(encode_beatmap(bm, get_frame_times(audio.size(-1)))).float()
        labels = th.tensor(get_labels(bm)).float()
            
        yield Batch(audio,chart,labels)
        
class SubsequenceDataset(FullSequenceDataset):
    def sample_map(self, map_file: Path, map_idx: int):
        for audio,chart,labels in super().sample_map(map_file, map_idx):
            L = audio.size(-1)
            if self.seq_len >= L:
                return

            num_samples = int(L / self.seq_len)
            for idx in th.randperm(L - self.seq_len)[:num_samples]:
                sl = ..., slice(idx,idx+self.seq_len)
                yield Batch(audio[sl], chart[sl], labels)
