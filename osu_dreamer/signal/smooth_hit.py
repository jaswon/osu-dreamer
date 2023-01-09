import numpy as np
import scipy

# std dev of impulse indicating a hit
HIT_SD = 3

def sigmoid(x):
    """1/(e^-x + 1)"""
    return np.exp(-np.logaddexp(-x, 0))

def encode_hit(sig, frame_times, i):
    z = (frame_times-i)/HIT_SD

    # hits are impulses
    # sig += 2 * np.exp(-.5 * z**2)

    # hits are flips
    sig *= 1 - 2 * sigmoid(z)

def encode_hold(sig, frame_times, i, j):
    sig += 2 * (
        sigmoid((frame_times - i) / HIT_SD)
        - sigmoid((frame_times - j) / HIT_SD)
    )

def _decode(sig):
    return scipy.signal.find_peaks(sig)[0].astype(int).tolist()

def decode_hit(sig):
    # return _decode(sig)
    sig_grad = np.gradient(sig)
    return sorted([
        *(scipy.signal.find_peaks(sig_grad)[0].astype(int)),
        *(scipy.signal.find_peaks(-sig_grad)[0].astype(int)),
    ])

def decode_hold(sig):
    sig_grad = np.gradient(sig)

    start_idxs = _decode(sig_grad)
    end_idxs = _decode(-sig_grad)

    # ensure that first start is before first end
    while len(start_idxs) and len(end_idxs) and start_idxs[0] >= end_idxs[0]:
        end_idxs.pop(0)

    # ensure that there is one end for every start
    if len(start_idxs) > len(end_idxs):
        start_idxs = start_idxs[:len(end_idxs) - len(start_idxs)]
    elif len(end_idxs) > len(start_idxs):
        end_idxs = end_idxs[:len(start_idxs) - len(end_idxs)]
    
    return start_idxs, end_idxs