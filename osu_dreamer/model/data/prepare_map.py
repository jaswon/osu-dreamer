
import time
from functools import partial
from pathlib import Path
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from osu_dreamer.osu.beatmap import Beatmap

from .beatmap.encode import encode_beatmap
from .load_audio import load_audio, get_frame_times
from .reclaim_memory import reclaim_memory


def generate_dataset(maps_path: Path, dataset_path: Path, num_workers: int):
    src_maps = list(maps_path.rglob("*.osu"))
    num_src_maps = len(src_maps)
    if num_src_maps == 0:
        raise RuntimeError(f"no osu! beatmaps found in {maps_path}")
    
    print(f"{num_src_maps} osu! beatmaps found, processing...")
    with Pool(processes=num_workers) as p:
        for _ in tqdm(p.imap_unordered(partial(prepare_map, dataset_path), src_maps), total=num_src_maps):
            reclaim_memory()

def prepare_map(data_dir: Path, map_file: Path):
    try:
        bm = Beatmap(map_file, meta_only=True)
    except Exception as e:
        print(f"{map_file}: {e}")
        return

    if bm.mode != 0:
        # not osu!std, skip
        # print(f"{map_file}: not an osu!std map")
        return

    af_dir = "_".join([bm.audio_filename.stem, *(s[1:] for s in bm.audio_filename.suffixes)])
    map_dir = data_dir / map_file.parent.name / af_dir
    
    spec_path =  map_dir / "spec.pt"
    map_path = map_dir / f"{map_file.stem}.map.pt"
    
    if map_path.exists():
        return
    
    try:
        bm.parse_map_data()
    except Exception as e:
        print(f"{map_file}: {e}")
        return

    if spec_path.exists():
        for i in range(5):
            try:
                spec = np.load(spec_path)
                break
            except (ValueError, EOFError):
                # can be raised if file was created but writing hasn't completed
                # just wait a little for the writing to finish
                time.sleep(.001 * 2**i)
        else:
            # retried 5 times without success, just skip
            print(f"{bm.audio_filename}: unable to load spectrogram from {spec_path}")
            return
    else:
        # load audio file
        try:
            spec = load_audio(bm.audio_filename)
        except Exception as e:
            print(f"{bm.audio_filename}: {e}")
            return

        # save spectrogram
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        with open(spec_path, "wb") as f:
            np.save(f, spec, allow_pickle=False)
            
    frame_times = get_frame_times(spec)

    # compute map signal
    try:
        x = encode_beatmap(bm, frame_times)
    except Exception as e:
        print(e)
        raise RuntimeError('failed to encode beatmap')

    with open(map_path, "wb") as f:
        np.save(f, x, allow_pickle=False)