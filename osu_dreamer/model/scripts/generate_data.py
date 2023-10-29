
from pathlib import Path

import click

from ..data.prepare_map import generate_dataset

dir_option_type = click.Path(exists=True, file_okay=False, path_type=Path)

@click.command()
@click.option('--maps-dir', type=dir_option_type, required=True, help='directory containing uncompressed osu! mapsets (eg. the `osu!/Songs` directory)')
@click.option('--data-dir', type=dir_option_type, default='./data', help='directory to store pre-processed training samples')
@click.option('--num-workers', type=int, default=2, help='number of workers to use for dataset generation')
def generate_data(maps_path: Path, dataset_path: Path, num_workers: int):
    """
    generate training dataset from an `osu!/Songs` directory.
    
    this step is required for model training
    """
    generate_dataset(maps_path, dataset_path, num_workers)