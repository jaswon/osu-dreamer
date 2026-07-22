
from typing import Iterator, NamedTuple
from jaxtyping import Float
from torch import Tensor

from pathlib import Path
import random

import numpy as np
import torch as th
from torch.utils.data import IterableDataset, DataLoader

import pytorch_lightning as pl

from osu_dreamer.data.reclaim_memory import reclaim_memory
from osu_dreamer.data.beatmap.encode import NUM_LABELS

from .beatmap import hold_out_mapsets


class LatentBatch(NamedTuple):
    h: Float[Tensor, "A l"]         # audio features (chunk rate)
    z: Float[Tensor, "E l"]         # chart latent (chunk rate)
    s: Float[Tensor, "S"]           # per-map style code
    labels: Float[Tensor, str(f"{NUM_LABELS}")]


class LatentDataModule(pl.LightningDataModule):
    """serves cached latent encodings produced by `encode-latents`."""

    def __init__(
        self,
        batch_size: int,
        seq_len: int, # in latent frames
        num_workers: int,
        max_val_count: int = 512,
        max_val_frac: float = .3,
        data_path: str = "./data",
        shuffle_buffer_size: int = 1,
        max_per_map: int = -1,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_workers = num_workers
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_per_map = max_per_map

        data_dir = Path(data_path)
        val_sets = hold_out_mapsets(data_dir, '*.latent.npz', max_val_count, max_val_frac)
        self.train_set = LatentDataset(data_dir, dict(exclude=val_sets), self.seq_len, self.shuffle_buffer_size, self.max_per_map)
        self.val_set = LatentDataset(data_dir, dict(include=val_sets))

    def train_dataloader(self):
        return DataLoader(
            self.train_set, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            pin_memory=True, 
            persistent_workers=self.num_workers>0, 
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set, 
            batch_size=1, 
            num_workers=min(1, self.num_workers), 
            pin_memory=True, 
            persistent_workers=self.num_workers>0,
        )


def load_latents(latent_file: Path) -> LatentBatch:
    with np.load(latent_file) as d:
        z = th.from_numpy(d['z']).float()
        s = th.from_numpy(d['s']).float()
        labels = th.from_numpy(d['labels']).float()
    h = th.from_numpy(np.load(latent_file.parent / 'h.npy')).float()
    return LatentBatch(h, z, s, labels)


class LatentDataset(IterableDataset):
    def __init__(
        self,
        data_dir: Path, mapsets: dict[str, set[str]],
        seq_len: int | None = None, # None: yield full maps
        shuffle_buffer_size: int = 1,
        max_per_map: int = -1,
    ):
        super().__init__()
        if 'include' in mapsets:
            filter_fn = lambda sample: sample.parent.name in mapsets['include']
        elif 'exclude' in mapsets:
            filter_fn = lambda sample: sample.parent.name not in mapsets['exclude']
        else:
            raise ValueError('neither include nor exclude provided')
        self.dataset = list(filter(filter_fn, data_dir.rglob("*.latent.npz")))

        self.seq_len = seq_len
        self.shuffle_buffer_size = shuffle_buffer_size
        self.max_per_map = max_per_map if max_per_map > 0 else float('inf')

    def _sample_stream(self, num_workers: int, worker_id: int) -> Iterator[LatentBatch]:
        for i, map_file in enumerate(self.dataset):
            if i % num_workers != worker_id:
                continue
            try:
                yield from self.make_samples(map_file)
            finally:
                reclaim_memory()

    def __iter__(self):
        worker_info = th.utils.data.get_worker_info() # type: ignore
        if worker_info is None:
            num_workers, worker_id, seed = 1, 0, th.initial_seed()
        else:
            num_workers, worker_id, seed = worker_info.num_workers, worker_info.id, worker_info.seed

        random.seed(seed)

        stream = self._sample_stream(num_workers, worker_id)
        if self.shuffle_buffer_size <= 1:
            yield from stream
            return

        buffer: list[LatentBatch] = []
        for sample in stream:
            if len(buffer) < self.shuffle_buffer_size:
                buffer.append(sample)
                continue
            j = random.randrange(len(buffer))
            yield buffer[j]
            buffer[j] = sample
        random.shuffle(buffer)
        yield from buffer

    def make_samples(self, latent_file: Path) -> Iterator[LatentBatch]:
        h, z, s, labels = load_latents(latent_file)
        if self.seq_len is None:
            yield LatentBatch(h, z, s, labels)
            return

        offset_end = z.size(-1) - self.seq_len + 1
        if offset_end < 1:
            return
        offset_start = int(th.randint(0, min(self.seq_len, offset_end), ()).item())
        idxs = th.arange(offset_start, offset_end, self.seq_len)
        idxs = idxs[th.randperm(len(idxs))[:min(self.max_per_map, len(idxs))]] # type: ignore
        for i in idxs:
            yield LatentBatch(
                h[..., i:i+self.seq_len].clone(),
                z[..., i:i+self.seq_len].clone(),
                s, labels,
            )