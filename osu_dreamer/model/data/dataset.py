
from typing import NamedTuple
from torch import Tensor
from jaxtyping import Float, Int

import random
from collections.abc import Iterator

import numpy as np

import torch as th
from torch.utils.data import IterableDataset

from .reclaim_memory import reclaim_memory
from .load_audio import A_DIM
from .beatmap.encode import X_DIM


class Batch(NamedTuple):
    a: Float[Tensor, str(f"{A_DIM} L")]
    x: Float[Tensor, str(f"{X_DIM} L")]
    p: Int[Tensor, "L"]


class FullSequenceDataset(IterableDataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.dataset = kwargs.pop("dataset")
        self.sample_density = kwargs.pop("sample_density", 1.)
        
        if not 0 < self.sample_density <= 1:
            raise ValueError("sample density must be in (0, 1]:", self.sample_density)
            
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
        for i, sample in random.sample(list(enumerate(dataset)), int(len(dataset) * self.sample_density)):
            if i % num_workers != worker_id:
                continue
                
            try:
                for x in self.sample_stream(sample, i):
                    yield x
            finally:
                reclaim_memory()
            
    def sample_stream(self, map_file, map_idx) -> Iterator[Batch]:
        a = th.tensor(np.load(map_file.parent / "spec.pt")).float() # [A,L]
        with open(map_file, 'rb') as f:
            x = th.tensor(np.load(f)).float()
        
        yield Batch(a,x,th.arange(a.size(-1)))
        
        
class SubsequenceDataset(FullSequenceDataset):
    def __init__(self, **kwargs):
        self.seq_len = kwargs.pop("seq_len")
        self.subseq_density = kwargs.pop("subseq_density", 2)
        super().__init__(**kwargs)

        num_samples = 0
        for map_file in self.dataset:
            with open(map_file, 'rb') as f:
                magic = np.lib.format.read_magic(f)
                read_header = np.lib.format.read_array_header_1_0 if magic[0] == 1 else np.lib.format.read_array_header_2_0
                shape = read_header(f, max_header_size=100000)[0] # type: ignore
                num_samples += int(shape[-1] / self.seq_len * self.subseq_density)
        
        self.approx_dataset_size = num_samples * self.sample_density


    def sample_stream(self, map_file, map_idx):
        a,x,p = next(super().sample_stream(map_file, map_idx))
        L = a.size(-1)
        if self.seq_len >= L:
            return

        num_samples = int(L / self.seq_len * self.subseq_density)

        for idx in th.randperm(L - self.seq_len)[:num_samples]:
            sl = ..., slice(idx,idx+self.seq_len)
            yield Batch(a=a[sl], x=x[sl], p=p[sl]) 
