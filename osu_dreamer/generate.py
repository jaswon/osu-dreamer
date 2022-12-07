import random
import copy
from pathlib import Path

import numpy as np
import torch
import librosa
import scipy.stats

from osu_dreamer.signal.from_beatmap import timing_signal as beatmap_timing_signal
from osu_dreamer.model.data import load_audio, SR, HOP_LEN, N_FFT

def generate_mapset(
    model,
    audio_file,
    timing,
    num_samples,
    title,
    artist,
):
    from zipfile import ZipFile
    from osu_dreamer.signal import to_beatmap as signal_to_map
    
    metadata = dict(
        audio_filename=audio_file.name,
        title=title,
        artist=artist,
    )
    
    # load audio
    # ======
    dev = next(model.parameters()).device
    a, sr = load_audio(audio_file)
    a = torch.tensor(a, device=dev)

    frame_times = librosa.frames_to_time(
        np.arange(a.shape[-1]),
        sr=SR, hop_length=HOP_LEN, n_fft=N_FFT,
    ) * 1000
    
    # generate maps
    # ======
    
    # `timing` can be one of:
    # - List[TimingPoint] : timed according to timing points
    # - number : audio is constant known BPM
    # - None : no prior knowledge of audio timing
    if isinstance(timing, list):
        t = torch.tensor(beatmap_timing_signal(timing, frame_times), device=dev).float()
    else:
        if timing is None:
            bpm_prior = scipy.stats.lognorm(loc=np.log(180), scale=180, s=1)
        else:
            bpm_prior = scipy.stats.norm(loc=timing, scale=1)
            
        t = torch.tensor(librosa.beat.plp(
            onset_envelope=librosa.onset.onset_strength(
                S=a.cpu().numpy(), center=False,
            ),
            prior = bpm_prior,
            # use 10s of audio to determine local bpm
            win_length=int(10. * SR / HOP_LEN), 
        )[None], device=dev)
        
    
    pred_signals = model(
        a.repeat(num_samples,1,1),
        t[None, :].repeat(num_samples,1,1),
    ).cpu().numpy()

    random_hex_string = lambda num: hex(random.randrange(16**num))[2:]
    
    # package mapset
    # ======
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
                    pred_signal, frame_times, copy.deepcopy(timing),
                ),
            )
                
    return mapset