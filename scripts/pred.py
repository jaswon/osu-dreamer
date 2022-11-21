import random
import shutil

from pathlib import Path

import torch
import torch.nn.functional as F

from osu_dreamer.model import Model, load_audio, N_FFT
from osu_dreamer.osu.beatmap import Beatmap

VALID_PAD = 2048

def generate_mapset(audio_file, model_path, sample_steps, num_samples):
    use_cuda = torch.cuda.is_available()

    model = Model.load_from_checkpoint(model_path,
        sample_steps=sample_steps,
    )
    
    if use_cuda:
          model = model.cuda()
    model.eval()
    model.freeze()

    a, hop_length, sr = load_audio(audio_file)

    a = F.pad(torch.tensor(a), (VALID_PAD, VALID_PAD))

    pad = (1 + a.size(-1) // 2 ** model.depth) * 2 ** model.depth - a.size(-1)
    a = F.pad(a, (0, pad), mode='reflect')

    if use_cuda:
        a = a.cuda()

    pred = model(a.repeat(num_samples,1,1))[..., VALID_PAD:-VALID_PAD].cpu().numpy()

    # generate random name
    while True:
        mapset_dir = Path(f"{int(1e17*random.random()):x}")
        try:
            mapset_dir.mkdir()
            break
        except:
            pass

    shutil.copy(audio_file, mapset_dir / audio_file.name)

    for i, p in enumerate(pred):
        name = f"osu!dreamer {i}"
        pred_map = Beatmap.signal_to_map(audio_file, p, hop_length, N_FFT, sr, name=name)

        out_file = mapset_dir / f"{name}.osu"

        with open(out_file, "w") as f:
            f.write(pred_map)


    mapset_zip = Path(shutil.make_archive(mapset_dir, "zip", root_dir=mapset_dir))
    shutil.rmtree(mapset_dir)

    mapset = mapset_zip.with_suffix('.osz')
    mapset_zip.rename(mapset)
    return mapset
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='generate maps')
    parser.add_argument('audio_file', metavar='AUDIO_FILE', type=Path, help='audio file to map')
    parser.add_argument('model_path', metavar='MODEL_PATH', type=Path, help='path to the trained model')
    parser.add_argument('-S', dest='sample_steps', type=int, default=128, help='number of steps to sample')
    parser.add_argument('-N', dest='num_samples', type=int, default=3, help='number of maps to generate')

    args = parser.parse_args()
    generate_mapset(args.audio_file, args.model_path, args.sample_steps, args.num_samples)