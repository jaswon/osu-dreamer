
import warnings

from jaxtyping import Float

import numpy as np

import librosa

# audio processing constants
SR = 22000
MS_PER_FRAME = 8
HOP_LEN = (SR // 1000) * MS_PER_FRAME

N_OCTAVES = 8
OCTAVE_BINS = 12
A_DIM = N_OCTAVES * OCTAVE_BINS

FrameTimes = Float[np.ndarray, "L"]

def get_frame_times(spec) -> FrameTimes:
    return librosa.frames_to_time(
        np.arange(spec.shape[-1]),
        sr=SR, hop_length=HOP_LEN,
    ) * 1000

eps = 1e-10
fmin = librosa.note_to_hz('C0')

def load_audio(audio_file):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wave, _ = librosa.load(audio_file, sr=SR, res_type='polyphase')

    # compute time-frequency representation
    return np.log(eps + np.abs(librosa.vqt(
        y=wave,
        sr=SR,
        hop_length=HOP_LEN,
        fmin=fmin,
        n_bins=A_DIM,
        bins_per_octave=OCTAVE_BINS,
    )))