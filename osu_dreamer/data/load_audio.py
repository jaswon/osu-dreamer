
from typing import Optional, Union
from jaxtyping import Float

from subprocess import CalledProcessError, run

import numpy as np

import torch as th
from torch import Tensor

import nnAudio.features

# audio processing constants
F_MIN = 32 # ~C1
BINS_PER_OCTAVE = 9
N_OCTAVES = 8
A_DIM = BINS_PER_OCTAVE * N_OCTAVES
SR = F_MIN * 1 << (1 + N_OCTAVES)
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

def load_audio(
    file_name, 
    device: Optional[Union[str, th.device]] = None,
) -> Float[np.ndarray, "F L"]:
    try:
        buf = run([
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", file_name,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(SR),
            "-",
        ], capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    
    wave = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 2 ** 15
    wave = th.from_numpy(wave).float()
    if device is not None:
        wave = wave.to(device)

    # spec = mel_spectrogram(wave)
    spec = constant_q_transform(wave)

    log_spec = th.clamp(spec, min=1e-10).log10()
    log_spec = th.maximum(log_spec, log_spec.max() - 6.) / 3
    return log_spec.numpy()

cqt_module = nnAudio.features.CQT(
    sr=SR,
    hop_length=HOP_LEN,
    fmin=F_MIN,
    n_bins=A_DIM,
    bins_per_octave=BINS_PER_OCTAVE,
    verbose=False,
    output_format='Complex',
)

def constant_q_transform(wave: Float[Tensor, "N"]) -> Float[Tensor, f'{A_DIM} L']:
    cqt = cqt_module.to(wave.device)(wave)[0]
    return cqt.pow(2).sum(-1)

N_FFT = 512
def mel_spectrogram(wave: Float[Tensor, "N"]) -> Float[Tensor, f'{A_DIM} L']:
    stft = th.stft(
        wave, 
        n_fft=N_FFT, 
        hop_length=HOP_LEN, 
        window=th.hann_window(N_FFT).to(wave.device), 
        pad_mode='constant',
        return_complex=True,
    )
    spec = stft[..., :-1].abs().pow(2)
    return mel_filter_banks() @ spec

# mel scale (O'Shaughnessy 1987)
mel2hz = lambda m: 700 * np.expm1(m/1127)
hz2mel = lambda f: 1127 * np.log1p(f/700)

def mel_filter_banks() -> Float[Tensor, f"{A_DIM} {1 + N_FFT//2}"]:
    mel_f = mel2hz(np.linspace(hz2mel(0), hz2mel(SR / 2.), A_DIM+2))
    fdiff = np.diff(mel_f)[:,None] # M+1
    ramps = np.subtract.outer(mel_f, np.fft.rfftfreq(n=N_FFT, d=SR ** -1)) # M+2 F
    weights = np.maximum(0, np.minimum(-ramps[:-2] / fdiff[:-1], ramps[2:] / fdiff[1:]))

    # divide mel weights by band width (area normalization)
    weights *= 2.0 / (mel_f[2:] - mel_f[:-2])[:,None]

    return th.from_numpy(weights).float()