
from pathlib import Path
from functools import partial
from multiprocessing import Pool

from tqdm import tqdm

from osu_dreamer.data.reclaim_memory import reclaim_memory
from osu_dreamer.data.prepare_map import prepare_map

import click

dir_option_type = click.Path(exists=True, file_okay=False, path_type=Path)

@click.command()
@click.option('--maps-dir', type=dir_option_type, required=True, help='directory containing uncompressed osu! mapsets (eg. the `osu!/Songs` directory)')
@click.option('--data-dir', type=click.Path(), default=Path('./data'), help='directory to store pre-processed training samples')
@click.option('--num-workers', type=int, default=2, help='number of workers to use for dataset generation')
def generate_data(maps_dir: Path, data_dir: Path, num_workers: int):
    """
    generate training dataset from an `osu!/Songs` directory.
    
    this step is required for model training
    """
    src_maps = list(maps_dir.rglob("*.osu"))
    num_src_maps = len(src_maps)
    if num_src_maps == 0:
        raise RuntimeError(f"no osu! beatmaps found in {maps_dir}")
    
    print(f"{num_src_maps} osu! beatmaps found, processing...")

    data_dir.mkdir(exist_ok=True)
    with Pool(processes=num_workers) as p:
        for _ in tqdm(p.imap_unordered(partial(prepare_map, data_dir), src_maps), total=num_src_maps):
            reclaim_memory()