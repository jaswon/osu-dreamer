
from typing import Optional

from pathlib import Path
import random
from zipfile import ZipFile

import torch as th

import click

from ..data.load_audio import load_audio, get_frame_times
from ..data.beatmap.decode import decode_beatmap, Metadata

from ..model import Model
    

file_option_type = click.Path(exists=True, dir_okay=False, path_type=Path)

@click.command()
@click.option('--model-path',   type=file_option_type, required=True, help='trained model (.ckpt)')
@click.option('--audio-file',   type=file_option_type, required=True, help='audio file to map')
@click.option('--sample-steps', type=int, default=32, help='number of diffusion steps to sample')
@click.option('--num-samples',  type=int, default=1 , help='number of maps to generate')
@click.option('--title',        type=str, help='Song title - required if it cannot be determined from the audio metadata')
@click.option('--artist',       type=str, help='Song artist - required if it cannot be determined from the audio metadata')
def predict(
    model_path: Path,
    audio_file: Path,
    sample_steps: int,
    num_samples: int,
    title: Optional[str],
    artist: Optional[str],
):
    """generate osu!std maps from raw audio."""
    
    # read metadata from audio file
    # ======
    import mutagen
    tags: mutagen.FileType = mutagen.File(audio_file, easy=True) # type: ignore
    
    if title is None:
        try:
            title = tags['title'][0]
        except KeyError:
            raise ValueError('no title provided, and unable to determine title from audio metadata')
        
    if artist is None:
        try:
            artist = tags['artist'][0]
        except KeyError:
            raise ValueError('no artist provided, and unable to determine artist from audio metadata')
            
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
    a = th.tensor(load_audio(audio_file), device=dev)

    frame_times = get_frame_times(a)
    
    # generate maps
    # ======
    with th.no_grad():
        pred_signals = model.sample(a, num_samples, sample_steps, show_progress=True).cpu().numpy()

    # package mapset
    # ======
    random_hex_string = lambda num: hex(random.randrange(16**num))[2:]

    while True:
        mapset = Path(f"_{random_hex_string(7)} {artist} - {title}.osz")
        if not mapset.exists():
            break
            
    with ZipFile(mapset, 'x') as mapset_archive:
        mapset_archive.write(audio_file, audio_file.name)
        
        for i, pred_signal in enumerate(pred_signals):
            mapset_archive.writestr(
                f"{artist} - {title} (osu!dreamer) [version {i}].osu",
                decode_beatmap(
                    Metadata(audio_file.name, title, artist, f"version {i}"),
                    pred_signal, frame_times,
                ),
            )
    