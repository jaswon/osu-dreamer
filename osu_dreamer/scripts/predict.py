from typing import Optional
from collections.abc import Sequence

from pathlib import Path
import random
from zipfile import ZipFile

import torch as th

import click

from osu_dreamer.data.load_audio import load_audio
from osu_dreamer.lm.model import Model
from osu_dreamer.lm.data.tokens.decoder import Decoder
from osu_dreamer.lm.data.parse.beatmap import to_beatmap, BeatmapDifficulty, Metadata


file_option_type = click.Path(exists=True, dir_okay=False, path_type=Path)

@click.command()
@click.option('--model-path',   type=file_option_type, required=True, help='trained model (.ckpt)')
@click.option('--audio-file',   type=file_option_type, required=True, help='audio file to map')
@click.option('--diff',         type=(float, float, float, float, float), multiple=True, help='difficulty attributes (hp,cs,od,ar,slider tick rate)')
@click.option('--title',        type=str, help='Song title - required if it cannot be determined from the audio metadata')
@click.option('--artist',       type=str, help='Song artist - required if it cannot be determined from the audio metadata')
def predict(
    model_path: Path,
    audio_file: Path,
    diff: Sequence[tuple[float, float, float, float, float]],
    title: Optional[str],
    artist: Optional[str],
):
    """generate osu!std maps from raw audio using the language model."""
    
    # read metadata from audio file
    # ======
    from tinytag import TinyTag
    tags = TinyTag.get(audio_file)
    assert isinstance(tags, TinyTag)
    
    if title is None:
        if tags.title is None:
            raise ValueError('no title provided, and unable to determine title from audio metadata')
        title = tags.title
        
    if artist is None:
        if tags.artist is None:
            raise ValueError('no artist provided, and unable to determine artist from audio metadata')
        artist = tags.artist
            
    # load model
    # ======
    model = Model.load_from_checkpoint(model_path).eval()
    
    if th.cuda.is_available():
        print('using GPU accelerated inference')
        model = model.cuda()
    else:
        print('WARNING: no GPU found - inference will be slow')
    
    # load audio
    # ======
    dev = next(model.parameters()).device
    audio = th.tensor(load_audio(audio_file), device=dev).float()
    
    # generate maps
    # ======
    maps = []
    for i, d in enumerate(diff):
        print(f"generating map for difficulty {i+1}/{len(diff)}")
        map_features = th.tensor(d, device=dev).float()
        
        with th.no_grad():
            pred_seq = model.sample(
                audio, map_features,
                max_len=-1,
                show_progress=True,
            )

        map_events = Decoder(model.vocab, pred_seq).parse_beatmap_events()
        map_diff = BeatmapDifficulty(
            hp_drain_rate=d[0],
            circle_size=d[1],
            overall_difficulty=d[2],
            approach_rate=d[3],
            slider_tick_rate=d[4],
        )

        maps.append((map_events, map_diff))

    # package mapset
    # ======
    random_hex_string = lambda num: hex(random.randrange(16**num))[2:]

    while True:
        mapset = Path(f"_{random_hex_string(7)} {artist} - {title}.osz")
        if not mapset.exists():
            break
            
    with ZipFile(mapset, 'x') as mapset_archive:
        mapset_archive.write(audio_file, audio_file.name)
        
        for i, (map_events, map_diff) in enumerate(maps):
            mapset_archive.writestr(
                f"{artist} - {title} (osu!dreamer) [version {i}].osu",
                to_beatmap(
                    map_events, map_diff, 
                    Metadata(audio_file.name, title, artist, f"version {i}"),
                ),
            )
    