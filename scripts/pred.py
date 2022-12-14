from pathlib import Path

import torch

from osu_dreamer.osu.hit_objects import TimingPoint
from osu_dreamer.osu.beatmap import Beatmap
from osu_dreamer.model import Model
from osu_dreamer import generate_mapset 

def list_of_timing_points(s):
    if not s:
        return None
    
    l = []
    for i, u in enumerate(s.split("|")):
        us = u.split(":")
        if len(us) != 3:
            raise ValueError(f"{i}: timing points must be formatted like `OFFSET:BEAT_LENGTH:METER`, got `{u}`")
            
        for j, f in enumerate(['OFFSET', 'BEAT_LENGTH', 'METER']):
            try:
                us[j] = float(us[j])
            except:
                raise ValueError(f"{i}: `{f}` must be a number, got {us[j]}")
            
        l.append(TimingPoint(int(us[0]), us[1], None, int(us[2])))
    
    return l
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='generate osu!std maps from raw audio')
    parser.add_argument('model_path', metavar='MODEL_PATH', type=Path, help='trained model (.ckpt)')
    parser.add_argument('audio_file', metavar='AUDIO_FILE', type=Path, help='audio file to map')
    
    model_args = parser.add_argument_group('model arguments')
    model_args.add_argument('--sample_steps', type=int, default=128, help='number of steps to sample')
    model_args.add_argument('--num_samples', type=int, default=3, help='number of maps to generate')
    
    timing_args = parser.add_argument_group('timing arguments')
    timing_args.add_argument('--bpm', type=float,
        help='tempo of the whole song in BPM (optional)')
    timing_args.add_argument('--timing_points_from', type=Beatmap,
        help='beatmap file to take timing points from (optional)')
    timing_args.add_argument('--timing_points', type=list_of_timing_points,
        help='list of pipe-separated timing points in `OFFSET:BEAT_LENGTH:METER` format (optional)')
    
    metadata_args = parser.add_argument_group('metadata arguments')
    metadata_args.add_argument('--title',
        help='Song title - required if it cannot be determined from the audio metadata')
    metadata_args.add_argument('--artist',
        help='Song artsit - required if it cannot be determined from the audio metadata')
    
    args = parser.parse_args()
    
    # read metadata from audio file
    # ======
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

    timing = None
    if args.timing_points_from is not None:
        timing = args.timing_points_from.uninherited_timing_points
    elif args.timing_points is not None:
        timing = args.timing_points
    elif args.bpm is not None:
        timing = args.bpm
            
    # load model
    # ======
    model = Model.load_from_checkpoint(
        args.model_path,
        sample_steps=args.sample_steps,
    ).eval()
    
    if torch.cuda.is_available():
        print('using GPU accelerated inference')
        model = model.cuda()
    else:
        print('WARNING: no GPU found - inference will be slow')
    
    generate_mapset(
        model,
        args.audio_file,
        timing,
        args.num_samples,
        args.title, args.artist,
    )