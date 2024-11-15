
from pathlib import Path
from multiprocessing import Pool

import numpy as np

from tqdm import tqdm

from osu_dreamer.osu.beatmap import Beatmap
from osu_dreamer.data.reclaim_memory import reclaim_memory
from osu_dreamer.data.beatmap.encode import encode_beatmap
from osu_dreamer.data.labels import get_labels
from osu_dreamer.data.load_audio import load_audio, get_frame_times

import click

dir_option_type = click.Path(exists=True, file_okay=False, path_type=Path)

@click.command()
@click.option('--maps-dir', type=dir_option_type, required=True, help='directory containing uncompressed osu! mapsets (eg. the `osu!/Songs` directory)')
@click.option('--data-dir', type=click.Path(path_type=Path), default=Path('./data'), help='directory to store pre-processed training samples')
@click.option('--num-workers', type=click.IntRange(min=1), default=2, help='number of workers to use for dataset generation')
def generate_data(maps_dir: Path, data_dir: Path, num_workers: int):
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
    with Pool(processes=num_workers) as p:
        for _ in tqdm(p.imap_unordered(process_mapset, src_mapsets.items()), total=len(src_mapsets)):
            reclaim_memory()

def process_mapset(kv: tuple[Path, list[Path]]):
    mapset_dir, map_files = kv
    audio_map: dict[tuple[Path, Path], list[tuple[Beatmap, Path]]] = {}
    for map_file in map_files:
        try:
            bm = Beatmap(map_file, meta_only=True)
        except Exception as e:
            print(f"{map_file}: {e}")
            continue

        if bm.mode != 0:
            # not osu!std, skip
            # print(f"{map_file}: not an osu!std map")
            continue

        audio_dir = mapset_dir / "_".join(bm.audio_filename.name.split('.'))
        map_path = audio_dir / f"{map_file.stem}.map.pt"
        if map_path.exists():
            continue

        audio_map.setdefault((audio_dir, bm.audio_filename), []).append((bm, map_path))

    for (audio_dir, audio_file), bms in audio_map.items():
        try:
            spec = load_audio(audio_file)
        except Exception as e:
            print(f"{audio_file}: {e}")
            return
        
        # save spectrogram
        spec_path = audio_dir / "spec.pt"
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        with open(spec_path, "wb") as f:
            np.save(f, spec, allow_pickle=False)

        for bm, map_path in bms:
            frame_times = get_frame_times(spec.shape[1])
            
            try:
                bm.parse_map_data()
            except Exception as e:
                print(f"{map_file}: {e}")
                continue
            diff_labels = get_labels(map_file, bm)

            # compute map signal
            try:
                x = encode_beatmap(bm, frame_times)
            except Exception as e:
                print(f"{map_file} [encode]:{e}")
                continue

            with open(map_path, "wb") as f:
                for obj in [x, diff_labels]:
                    np.save(f, obj, allow_pickle=False)
        