
from jaxtyping import Float

import numpy as np

import torch as th
import nnAudio.features
import ffmpeg

# audio processing constants
F_MIN = 128 # ~C3
BINS_PER_OCTAVE = 12
N_OCTAVES = 6
A_DIM = BINS_PER_OCTAVE * N_OCTAVES
SR = F_MIN * 1 << (1 + N_OCTAVES)
MS_PER_FRAME = 6 # approximate
HOP_LEN = (SR * MS_PER_FRAME + 500) // 1000
eps = 1e-10

FrameTimes = Float[np.ndarray, "L"]

def get_frame_times(num_frames: int) -> FrameTimes:
    """returns the time (ms) corresponding to frames"""
    frames = np.arange(num_frames)
    samples = frames * HOP_LEN
    return samples / SR * 1000

def load_audio(file_name):
    buf = ( ffmpeg
        .input(file_name)
        .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=str(SR))
        .overwrite_output()
        .run(quiet=True)
    )[0]
    wave = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 2 ** 15

    dev = 'cuda' if th.cuda.is_available() else 'cpu'
    cqt = nnAudio.features.CQT(
        sr=SR,
        hop_length=HOP_LEN,
        fmin=F_MIN,
        n_bins=A_DIM,
        bins_per_octave=BINS_PER_OCTAVE,
        verbose=False,
    ).to(dev)(th.tensor(wave, device=dev))[0].cpu().numpy()
    return np.log(eps + cqt**2) - np.log(eps)