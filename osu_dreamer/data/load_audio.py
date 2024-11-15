
from jaxtyping import Float

import numpy as np
from numpy import ndarray

import ffmpeg

# audio processing constants
A_DIM = 96
SR = 22000
MS_PER_FRAME = 8
HOP_LEN = (SR // 1000) * MS_PER_FRAME
N_FFT = 2048
eps = 1e-10

FrameTimes = Float[np.ndarray, "L"]

def get_frame_times(num_frames: int) -> FrameTimes:
    """returns the time (ms) corresponding to frames"""
    frames = np.arange(num_frames)
    samples = frames * HOP_LEN + N_FFT // 2
    return samples / SR * 1000

def load_audio(file_name):
    return compute_repr(read_file(file_name))

def read_file(file_name) -> Float[ndarray, "N"]:
    buf = ( ffmpeg
        .input(file_name)
        .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar=str(SR))
        .overwrite_output()
        .run(quiet=True)
    )[0]
    return np.frombuffer(buf, dtype=np.int16).astype(np.float32) / float(1 << 15)

def compute_repr(wave: Float[ndarray, "N"]) -> Float[ndarray, "M L"]:
    try:
        import librosa
        mel = librosa.feature.melspectrogram(
            y=wave, 
            sr=SR, 
            n_fft=N_FFT, 
            hop_length=HOP_LEN, 
            n_mels=A_DIM, 
            htk=True,
        )
    except ImportError:
        mel = compute_scipy_mel(wave)
    return np.log(eps + mel)
    
def compute_scipy_mel(wave: Float[ndarray, "N"]) -> Float[ndarray, "M L"]:
    import scipy.signal
    from scipy.signal.windows import hann

    S = scipy.signal.ShortTimeFFT(hann(N_FFT, sym=False), hop=HOP_LEN, fs=SR)
    spec = S.spectrogram(wave)[:,S.lower_border_end[1]-1:S.p_min-1]
    return np.einsum('fl,mf->ml', spec, mel_filter_banks(S.f))

# mel scale (O'Shaughnessy 1987)
mel2hz = lambda m: 700 * np.expm1(m/1127)
hz2mel = lambda f: 1127 * np.log1p(f/700)

def mel_filter_banks(fft_freqs: Float[ndarray, "F"]) -> Float[ndarray, "M F"]:
    mel_f = mel2hz(np.linspace(hz2mel(0), hz2mel(SR / 2.), A_DIM+2))

    fdiff = np.diff(mel_f)[:,None] # M+1
    ramps = np.subtract.outer(mel_f, fft_freqs) # M+2 F
    weights = np.maximum(0, np.minimum(-ramps[:-2] / fdiff[:-1], ramps[2:] / fdiff[1:]))

    # divide mel weights by band width (area normalization)
    weights *= 2.0 / (mel_f[2:] - mel_f[:-2])[:,None]

    return weights