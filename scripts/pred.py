import random
import shutil

from pathlib import Path

import torch
import torch.nn.functional as F

from osu_dreamer.model import Model, load_audio, N_FFT
from osu_dreamer.osu.beatmap import Beatmap

def random_hex_string(num):
    import random
    return hex(random.randrange(16**num))[2:]

def generate_mapset(
    audio_file,
    model_path,
    sample_steps,
    num_samples,
    title,
    artist,
    bpm = None,
):
    
    # check for GPU
    # ======
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('using GPU accelerated inference')
    else:
        print('WARNING: no GPU found - inference will be slow')

    # load model
    # ======
    model = Model.load_from_checkpoint(model_path,
        sample_steps=sample_steps,
    )
    
    if use_cuda:
          model = model.cuda()
    model.eval()
    
    # load audio
    # ======
    a, hop_length, sr = load_audio(audio_file)
    a = torch.tensor(a)

    if use_cuda:
        a = a.cuda()
        
    # generate maps
    # ======
    pred = model(a.repeat(num_samples,1,1)).cpu().numpy()

    # package mapset
    # ======
    while True:
        mapset_dir = Path(f"_{random_hex_string(7)} {artist} - {title}")
        try:
            mapset_dir.mkdir()
            break
        except:
            pass

    shutil.copy(audio_file, mapset_dir / audio_file.name)

    for i, p in enumerate(pred):
        pred_map = Beatmap.signal_to_map(
            dict(
                audio_filename=audio_file.name,
                title=title,
                artist=artist,
                version=f"version {i}",
            ),
            p, hop_length, N_FFT, sr, bpm=bpm,
        )

        out_file = mapset_dir / f"{artist} - {title} (osu!dreamer) [version {i}].osu"

        with open(out_file, "w") as f:
            f.write(pred_map)

    mapset_zip = Path(shutil.make_archive(mapset_dir, "zip", root_dir=mapset_dir))
    shutil.rmtree(mapset_dir)

    mapset = mapset_zip.with_suffix('.osz')
    mapset_zip.rename(mapset)
    return mapset
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='generate osu!std maps from raw audio')
    parser.add_argument('model_path', metavar='MODEL_PATH', type=Path, help='trained model (.ckpt)')
    parser.add_argument('audio_file', metavar='AUDIO_FILE', type=Path, help='audio file to map')
    
    model_args = parser.add_argument_group('model arguments')
    model_args.add_argument('--sample_steps', type=int, default=128, help='number of steps to sample')
    model_args.add_argument('--num_samples', type=int, default=3, help='number of maps to generate')
    model_args.add_argument('--bpm', type=int, help='BPM of audio (not required)')
    
    metadata_args = parser.add_argument_group('metadata arguments')
    metadata_args.add_argument('--title',
        help='Song title - must be provided if it cannot be determined from the audio metadata')
    metadata_args.add_argument('--artist',
        help='Song artsit - must be provided if it cannot be determined from the audio metadata')
    
    args = parser.parse_args()
    
    import mutagen
    tags = mutagen.File(args.audio_file, easy=True)
    
    if args.title is None:
        try:
            args.title = tags['title'][0]
        except KeyError:
            parser.error('no title provided, and unable to determine title from audio metadata')
        
    if args.artist is None:
        try:
            args.artist = tags['artist'][0]
        except KeyError:
            parser.error('no artist provided, and unable to determine artist from audio metadata')
    
    if args.bpm is None and 'bpm' in tags:
        args.bpm = float(tags['bpm'][0])
            
    generate_mapset(
        args.audio_file,
        args.model_path,
        args.sample_steps,
        args.num_samples,
        args.title,
        args.artist,
        args.bpm,
    )