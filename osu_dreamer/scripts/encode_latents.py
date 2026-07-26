
from pathlib import Path

import click
import numpy as np
import torch as th

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from osu_dreamer.data.modules.beatmap import pad_to_multiple
from osu_dreamer.data.load_audio import read_spec
from osu_dreamer.data.beatmap.encode import read_beatmap
from osu_dreamer.models.latent.train import LatentTrainer


class MapsetDataset(Dataset):
    """loads one mapset (shared spec + all difficulty charts) per item,
    entirely in worker processes so the GPU never waits on disk I/O."""

    def __init__(self, mapsets: list[tuple[Path, list[Path], bool]]):
        self.mapsets = mapsets

    def __len__(self):
        return len(self.mapsets)

    def __getitem__(self, idx: int):
        mapset_dir, map_files, need_h = self.mapsets[idx]

        spec = None
        if need_h:
            with open(mapset_dir / 'spec.npy', 'rb') as f:
                spec = read_spec(f)

        charts, labels = [], []
        for map_file in map_files:
            with open(map_file, 'rb') as f:
                chart_arr, label_arr = read_beatmap(f)
            charts.append(chart_arr)
            labels.append(label_arr)

        return mapset_dir, map_files, spec, charts, labels


@click.command()
@click.option('--latent-ckpt-path', type=click.Path(exists=True, dir_okay=False), default='latent.ckpt', help='path to the latent checkpoint')
@click.option(        '--data-dir', type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path('./data'), help='pre-processed dataset directory')
@click.option(          '--device', type=str, default=None, help='torch device (default: cuda if available)')
@click.option(     '--num-workers', type=int, default=4, help='dataloader workers for parallel file I/O')
@click.option(           '--force', is_flag=True, help='overwrite existing cached latents')
def encode_latents(latent_ckpt_path: str, data_dir: Path, device: str | None, num_workers: int, force: bool):
    """precompute latent-model encodings (h, z, s, labels) for diffusion training.

    caches, for every `*.map.npy`: `<map>.latent.npz` (z, s, labels), and per
    mapset directory: `h.npy` (audio features at chunk rate).
    """
    dev = th.device(device if device else ('cuda' if th.cuda.is_available() else 'cpu'))
    latent = LatentTrainer.load_from_checkpoint(latent_ckpt_path, map_location=dev).latent.eval().float().to(dev)
    c = latent.chunk_size

    map_files = sorted(data_dir.rglob("*.map.npy"))
    if len(map_files) == 0:
        raise RuntimeError(f'no pre-processed maps found in {data_dir}')

    # group by mapset: difficulties share audio, so charts can be batched
    by_mapset: dict[Path, list[Path]] = {}
    for map_file in map_files:
        by_mapset.setdefault(map_file.parent, []).append(map_file)

    def out_file_for(map_file: Path) -> Path:
        return map_file.with_name(map_file.name.removesuffix('.map.npy') + '.latent.npz')

    # skip work that is already cached
    mapsets: list[tuple[Path, list[Path], bool]] = []
    for mapset_dir, files in by_mapset.items():
        need_h = force or not (mapset_dir / 'h.npy').exists()
        pending = files if force else [f for f in files if not out_file_for(f).exists()]
        if need_h or pending:
            mapsets.append((mapset_dir, pending, need_h))

    loader = DataLoader(
        MapsetDataset(mapsets),
        batch_size=None,  # one mapset per item; batching over difficulties below
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=lambda x: x,
    )

    with th.inference_mode():
        for mapset_dir, mset_files, spec, charts, labels in tqdm(loader, total=len(mapsets)):
            if spec is not None:
                a = th.from_numpy(spec).float()[None].to(dev)
                _, h = latent.audio_encoder(pad_to_multiple(a, c))
                np.save(mapset_dir / 'h.npy', h[0].cpu().numpy())

            if len(mset_files) == 0:
                continue

            # batch all difficulties of the mapset in one forward pass
            # (process_sample guarantees all charts in a mapset share the same shape)
            x = th.stack([th.from_numpy(chart).float() for chart in charts]).to(dev)
            z, s = latent.encode_chart(pad_to_multiple(x, c))
            z, s = z.cpu().numpy(), s.cpu().numpy()
            for i, map_file in enumerate(mset_files):
                np.savez(out_file_for(map_file), z=z[i], s=s[i], labels=labels[i])
