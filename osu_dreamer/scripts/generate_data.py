
from pathlib import Path
import pickle
from functools import partial
from torch.multiprocessing import Pool, set_start_method

import numpy as np

from tqdm import tqdm

from osu_dreamer.lm.data.parse.parse_beatmap import from_beatmap, BeatmapDifficulty, BeatmapEvents
from osu_dreamer.lm.data.parse.parse_file import parse_map_file
from osu_dreamer.data.reclaim_memory import reclaim_memory
from osu_dreamer.data.load_audio import load_audio

import click

dir_option_type = click.Path(exists=True, file_okay=False, path_type=Path)

@click.command()
@click.option('--maps-dir', type=dir_option_type, required=True, help='directory containing uncompressed osu! mapsets (eg. the `osu!/Songs` directory)')
@click.option('--data-dir', type=click.Path(path_type=Path), default=Path('./data'), help='directory to store pre-processed training samples')
@click.option('--num-workers', type=click.IntRange(min=1), default=2, help='number of workers to use for dataset generation')
@click.option('--force', is_flag=True, help='whether to overwrite existing pre-processed maps')
def generate_data(maps_dir: Path, data_dir: Path, num_workers: int, force: bool):
    """
    generate training dataset from an `osu!/Songs` directory.
    
    this step is required for model training
    """
    src_maps = list(maps_dir.rglob("*.osu"))
    num_src_maps = len(src_maps)
    if num_src_maps == 0:
        raise RuntimeError(f"no osu! beatmaps found in {maps_dir}")

    src_mapsets = {}
    for src_map in src_maps:
        src_mapsets.setdefault(data_dir / src_map.parent.name, []).append(src_map)
    
    print(f"{num_src_maps} osu! beatmaps ({len(src_mapsets)} mapsets) found, processing...")

    data_dir.mkdir(exist_ok=True)
    set_start_method('spawn')
    with Pool(processes=num_workers) as p:
        for _ in tqdm(p.imap_unordered(partial(process_mapset, force=force), src_mapsets.items()), total=len(src_mapsets)):
            reclaim_memory()

def process_mapset(kv: tuple[Path, list[Path]], force: bool):
    mapset_dir, map_files = kv
    audio_map: dict[tuple[Path, Path], list[tuple[tuple[BeatmapEvents, BeatmapDifficulty], Path]]] = {}
    for map_file in map_files:
        try:
            with open(map_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            cfg = parse_map_file(lines)
            mode = int(cfg.general.get('Mode', '0'))
            if mode != 0:
                # not osu!std, skip
                continue

            events, diff, meta = from_beatmap(cfg)
            
            # account for case-insensitivity
            lc_files = { f.name.lower(): f.name for f in map_file.parent.iterdir() }
            audio_path = map_file.parent / lc_files[meta.audio_filename.lower()]
        
            audio_dir = mapset_dir / "_".join(audio_path.name.split('.'))
            map_path = audio_dir / f"{map_file.stem}.map.pkl"

            if not force and map_path.exists():
                continue

            data_to_pickle = (events, diff)
            audio_map.setdefault((audio_dir, audio_path), []).append((data_to_pickle, map_path))
        except Exception as e:
            raise Exception(map_file) from e

    for (audio_dir, audio_file), maps_to_save in audio_map.items():
        spec_path = audio_dir / "spec.pt"
        if not spec_path.exists():
            try:
                spec = load_audio(audio_file)
            except Exception as e:
                print(f"{audio_file}: {e}")
                return
            
            # save spectrogram
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            with open(spec_path, "wb") as f:
                np.save(f, spec, allow_pickle=False)

        for data_to_pickle, map_path in maps_to_save:
            with open(map_path, "wb") as f:
                pickle.dump(data_to_pickle, f)