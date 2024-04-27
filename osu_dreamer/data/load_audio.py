
import warnings

from jaxtyping import Float

import numpy as np

import librosa

# audio processing constants
N_FFT = 2048
SR = 22000
MS_PER_FRAME = 4
HOP_LEN = (SR // 1000) * MS_PER_FRAME
N_MELS = 64

A_DIM = 32

FrameTimes = Float[np.ndarray, "L"]

def get_frame_times(spec) -> FrameTimes:
    return librosa.frames_to_time(
        np.arange(spec.shape[-1]),
        sr=SR, hop_length=HOP_LEN, n_fft=N_FFT,
    ) * 1000

def load_audio(audio_file):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wave, _ = librosa.load(audio_file, sr=SR, res_type='polyphase')

    # compute spectrogram
    return librosa.feature.mfcc(
        y=wave,
        sr=SR,
        n_mfcc=A_DIM,
        n_fft=N_FFT,
        hop_length=HOP_LEN,
        n_mels=N_MELS,
    )