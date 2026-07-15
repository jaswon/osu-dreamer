
from pathlib import Path
from functools import partial

import click
from tqdm import tqdm

from torch.multiprocessing import Pool, Manager, set_start_method

from osu_dreamer.data.dataset import process_sample, make_dataset

dir_option_type = click.Path(exists=True, file_okay=False, path_type=Path)

@click.command()
@click.option('--data-dir', type=click.Path(path_type=Path), default=Path('./data'), help='directory to store pre-processed training samples')
@click.option('--num-workers', type=click.IntRange(min=1), default=2, help='number of workers to use for dataset generation')
@click.option('--force', is_flag=True, help='whether to overwrite existing pre-processed maps')
def generate_data(data_dir: Path, num_workers: int, force: bool):
    """
    generate training dataset.
    
    this step is required for model training
    """

    data_dir.mkdir(parents=True, exist_ok=True)
    dataset = make_dataset()
    if dataset.n_shards < num_workers:
        # not enough shards, cap workers
        num_workers = dataset.n_shards

    set_start_method('spawn', force=True)
    with Manager() as manager, Pool(processes=num_workers) as p:
        progress = manager.Queue()
        result = p.map_async(
            partial(process_dataset, force, data_dir, progress),
            [ dataset.shard(num_workers, i) for i in range(num_workers) ],
            callback=lambda _: progress.put(None),
            error_callback=lambda _: progress.put(None),
        )

        with tqdm(unit='sample') as pbar:
            while (item := progress.get()) is not None:
                pbar.update(item)
        result.get()  # re-raise worker exceptions

def process_dataset(force: bool, data_dir: Path, progress, dataset):
    for sample in dataset:
        process_sample(force, data_dir, sample)
        progress.put(1)