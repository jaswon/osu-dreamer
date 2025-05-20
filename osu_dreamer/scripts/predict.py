
from typing import Optional
from collections.abc import Sequence

from pathlib import Path
import random
from zipfile import ZipFile

import torch as th
import numpy as np

import click

from osu_dreamer.data.load_audio import load_audio

from osu_dreamer.decoder.data.parse import parse_tokens
from osu_dreamer.decoder.data.decode import decode, Metadata
from osu_dreamer.decoder.model import Model
    

file_option_type = click.Path(exists=True, dir_okay=False, path_type=Path)

@click.command()
@click.option('--model-path',   type=file_option_type, required=True, help='trained model (.ckpt)')
@click.option('--audio-file',   type=file_option_type, required=True, help='audio file to map')
@click.option('--diff',         type=(float, float, float, float, float), multiple=True, help='difficulty conditioning (sr, ar, od, cs, hp)')
@click.option('--sample-steps', type=int, default=32, help='number of diffusion steps to sample')
@click.option('--title',        type=str, help='Song title - required if it cannot be determined from the audio metadata')
@click.option('--artist',       type=str, help='Song artist - required if it cannot be determined from the audio metadata')
def predict(
    model_path: Path,
    audio_file: Path,
    diff: Sequence[tuple[float, float, float, float, float]],
    sample_steps: int,
    title: Optional[str],
    artist: Optional[str],
):
    """generate osu!std maps from raw audio."""
    
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
    labels = th.tensor(diff, device=dev)
    
    # generate maps
    # ======
    with th.autocast(device_type=dev.type), th.no_grad():
        pred_batch_tokens, pred_batch_labels = model.sample(audio, labels, show_progress=True)

    # package mapset
    # ======
    random_hex_string = lambda num: hex(random.randrange(16**num))[2:]

    while True:
        mapset = Path(f"_{random_hex_string(7)} {artist} - {title}.osz")
        if not mapset.exists():
            break
            
    with ZipFile(mapset, 'x') as mapset_archive:
        mapset_archive.write(audio_file, audio_file.name)
        
        for i, (pred_labels, pred_tokens) in enumerate(zip(pred_batch_labels, pred_batch_tokens)):
            mapset_archive.writestr(
                f"{artist} - {title} (osu!dreamer) [version {i}].osu",
                decode(
                    Metadata(audio_file.name, title, artist, f"version {i}"),
                    pred_labels, parse_tokens(pred_tokens, strict=False),
                ),
            )
    