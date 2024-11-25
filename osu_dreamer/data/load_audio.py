
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

FrameTimes = Float[np.ndarray, "L"]

def get_frame_times(num_frames: int) -> FrameTimes:
    """returns the time (ms) corresponding to frames"""
    frames = np.arange(num_frames)
    samples = frames * HOP_LEN
    return samples / SR * 1000

def load_audio(file_name) -> Float[np.ndarray, "F L"]:
    buf = ( ffmpeg
        .input(file_name)
        .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=str(SR))
        .overwrite_output()
        .run(quiet=True)
    )[0]
    wave = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 2 ** 15
    return compute_cqt(wave)

cqt_module = nnAudio.features.CQT(
    sr=SR,
    hop_length=HOP_LEN,
    fmin=F_MIN,
    n_bins=A_DIM,
    bins_per_octave=BINS_PER_OCTAVE,
    verbose=False,
)

def compute_cqt(wave, eps = 1e-3):
    dev = 'cuda' if th.cuda.is_available() else 'cpu'
    wave = th.tensor(wave, device=dev).float()
    cqt = cqt_module.to(dev)(wave, normalization_type='convolutional')[0].cpu().numpy()
    return (np.log(cqt+eps) - np.log(eps)) / -np.log(eps)