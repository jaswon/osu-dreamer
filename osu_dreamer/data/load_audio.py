
from typing import BinaryIO
from jaxtyping import Float

import numpy as np

from resonators import ResonatorBank # type: ignore

# audio processing constants
F_MIN = 32 # ~C1
BINS_PER_OCTAVE = 9
N_OCTAVES = 8
N_BINS = N_OCTAVES * BINS_PER_OCTAVE
A_DIM = BINS_PER_OCTAVE * N_OCTAVES
F_MAX = F_MIN * 1 << N_OCTAVES
SR = 2 * F_MAX
MS_PER_FRAME = 6 # approximate
HOP_LEN = (SR * MS_PER_FRAME + 500) // 1000

FrameTimes = Float[np.ndarray, "L"]

def get_frame_for_time(t_ms: int|float) -> int:
    """returns the frame index corresponding to t (ms)"""
    t_sec = t_ms / 1000
    sample = t_sec * SR
    frame = sample / HOP_LEN
    return int(frame)

def get_frame_times(num_frames: int) -> FrameTimes:
    """returns the time (ms) corresponding to frames"""
    frames = np.arange(num_frames)
    samples = frames * HOP_LEN
    return samples / SR * 1000

def make_spec(wave: Float[np.ndarray, "L"]) -> Float[np.ndarray, "F L"]:
    freqs = np.geomspace(F_MIN, F_MAX, N_BINS, endpoint=False).astype(np.float32)
    bank = ResonatorBank(freqs, SR)  # alphas default to a per-frequency heuristic
    spec = bank.resonate(wave, hop=HOP_LEN)  # shape (n_frames, n_bins), complex64

    sig = np.abs(spec.T) ** 2
    sig = np.maximum(1e-10, sig)
    sig = np.log10(sig) - np.log10(np.max(sig))
    sig = (15*sig+60)/60
    sig = np.clip(sig, a_min=0, a_max=1)
    return sig

### DISK FORMAT

SPEC_DTYPE = np.uint8

def write_spec(f: BinaryIO, spec: Float[np.ndarray, "F L"]):
    np.save(f, (spec * np.iinfo(SPEC_DTYPE).max + .5).astype(SPEC_DTYPE))

def read_spec(f: BinaryIO) -> Float[np.ndarray, "F L"]:
    return np.load(f).astype(float) / np.iinfo(SPEC_DTYPE).max