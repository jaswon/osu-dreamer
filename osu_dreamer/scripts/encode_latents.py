
from pathlib import Path

import click
import numpy as np
import torch as th

from tqdm import tqdm

from osu_dreamer.data.modules.beatmap import pad_to_multiple
from osu_dreamer.data.load_audio import read_spec
from osu_dreamer.data.beatmap.encode import read_beatmap
from osu_dreamer.models.latent.train import LatentTrainer

@click.command()
@click.option('--latent-ckpt-path', type=click.Path(exists=True, dir_okay=False), default='latent.ckpt', help='path to the latent checkpoint')
@click.option(        '--data-dir', type=click.Path(exists=True, file_okay=False, path_type=Path), default=Path('./data'), help='pre-processed dataset directory')
@click.option(          '--device', type=str, default=None, help='torch device (default: cuda if available)')
@click.option(           '--force', is_flag=True, help='overwrite existing cached latents')
def encode_latents(latent_ckpt_path: str, data_dir: Path, device: str | None, force: bool):
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

    with th.inference_mode():
        for map_file in tqdm(map_files):
            out_file = map_file.with_name(map_file.name.removesuffix('.map.npy') + '.latent.npz')
            h_file = map_file.parent / 'h.npy'
            if not force and out_file.exists() and h_file.exists():
                continue

            if force or not h_file.exists():
                with open(map_file.parent / 'spec.npy', 'rb') as f:
                    a = th.from_numpy(read_spec(f)).float()[None].to(dev)
                _, h = latent.audio_encoder(pad_to_multiple(a, c))
                np.save(h_file, h[0].cpu().numpy())

            with open(map_file, 'rb') as f:
                chart_arr, label_arr = read_beatmap(f)
            x = pad_to_multiple(th.from_numpy(chart_arr).float()[None].to(dev), c)
            z, s = latent.encode_chart(x)
            np.savez(out_file, z=z[0].cpu().numpy(), s=s[0].cpu().numpy(), labels=label_arr)
