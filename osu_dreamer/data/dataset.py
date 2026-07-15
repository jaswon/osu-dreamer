

from typing import TypedDict

from pathlib import Path

import traceback

import torchcodec
from datasets import IterableDataset, load_dataset, concatenate_datasets, Audio

from osu_dreamer.osu.beatmap import Beatmap, BeatmapParseError
from osu_dreamer.data.load_audio import SR, make_spec, write_spec, read_spec, get_frame_times
from osu_dreamer.data.beatmap.encode import write_beatmap

SampleBeatmap = TypedDict("SampleBeatmap", {
    "beatmapset_id": int,
    "beatmap_id": int,
    "mode": int,
    "approved": int,
    "content": str,
})

SampleJSON = TypedDict("SampleJSON", {
    "audio_hash": str,
    "beatmaps": list[SampleBeatmap],
})

Sample = TypedDict("Sample", {
    "json": SampleJSON,
    "opus": torchcodec.decoders._audio_decoder.AudioDecoder,
})

def make_dataset() -> IterableDataset:
    ds_dict = load_dataset("project-riz/osu-beatmaps", "compressed", streaming=True)
    dataset = concatenate_datasets(list(ds_dict.values()))
    dataset = dataset.cast_column("opus", Audio(sampling_rate=SR))
    return dataset

def process_sample(force: bool, data_dir: Path, sample: Sample):

    audio_hash = sample['json']['audio_hash']
    audio_dir = data_dir / audio_hash

    valid_bms: dict[Path, SampleBeatmap] = {}
    for bm in sample['json']['beatmaps']:
        if bm['mode'] != 0:
            # standard only
            continue

        if bm['approved'] != 1:
            # ranked only
            continue

        map_path = audio_dir / f"{bm['beatmap_id']}.map.npy"
        if map_path.exists() and not force:
            continue

        valid_bms[map_path] = bm

    if len(valid_bms) == 0:
        return
    
    spec_path = audio_dir / 'spec.npy'
    if spec_path.exists() and not force:
        with open(spec_path, 'rb') as f:
            spec = read_spec(f)
    else:
        try:
            wave = sample['opus'].get_all_samples().data.mean(dim=0).numpy()
            spec = make_spec(wave)
        except Exception as e:
            print(f"{audio_hash[:8]}... : {e}")
            return
            
        spec_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_spec = spec_path.with_suffix('.tmp')
        with open(tmp_spec, "wb") as f:
            write_spec(f, spec)
        tmp_spec.rename(spec_path)

    spec_frame_times = get_frame_times(spec.shape[1])
    for map_path, sample_bm in valid_bms.items():
        map_id = f"{sample_bm['beatmapset_id']}/{sample_bm['beatmap_id']}"

        try:
            bm = Beatmap(sample_bm['content'])
        except BeatmapParseError:
            continue
        except Exception as e:
            print()
            print(f'failed to parse beatmap {map_id}')
            traceback.print_exception(e)
            print()
            continue

        try:
            tmp_map = map_path.with_suffix('.tmp')
            with open(tmp_map, "wb") as f:
                write_beatmap(f, bm, spec_frame_times)
            tmp_map.rename(map_path)
        except Exception as e:
            print()
            print(f'failed to write beatmap {map_id}')
            traceback.print_exception(e)
            print()
            continue