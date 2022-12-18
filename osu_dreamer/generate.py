import random
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import librosa

from osu_dreamer.data import load_audio, HOP_LEN, SR, N_FFT
from osu_dreamer.signal import to_beatmap as signal_to_map

def generate_mapset(
    model,
    audio_file,
    timing,
    num_samples,
    title,
    artist,
    sample_steps,
    ddim,
):
    
    metadata = dict(
        audio_filename=audio_file.name,
        title=title,
        artist=artist,
    )
    
    # load audio
    # ======
    dev = next(model.parameters()).device
    a = torch.tensor(load_audio(audio_file), device=dev)

    frame_times = librosa.frames_to_time(
        np.arange(a.shape[-1]),
        hop_length=HOP_LEN,
        n_fft=N_FFT,
        sr=SR,
    ) * 1000
    
    # generate maps
    # ======
    pred_signals = model(
        a.repeat(num_samples,1,1),
        sample_steps=sample_steps,
        ddim=ddim,
    ).cpu().numpy()

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
                signal_to_map(
                    dict( **metadata, version=f"version {i}" ),
                    pred_signal, frame_times, timing,
                ),
            )
                
    return mapset