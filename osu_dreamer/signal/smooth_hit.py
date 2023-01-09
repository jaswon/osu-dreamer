import numpy as np
import scipy

# std dev of impulse indicating a hit
HIT_SD = 5

def encode_hit(sig, frame_times, i):
    z = (frame_times-i)/HIT_SD

    # hits are impulses
    # sig += np.exp(-.5 * z**2)

    # hits are flips
    sig *= 1 - 2 * np.exp(-np.logaddexp(-z, 0))

def encode_hold(sig, frame_times, i, j):
    z = np.where(frame_times<i,frame_times-i,np.where(frame_times<j,0,frame_times-j)) / HIT_SD
    sig += np.exp(-.5 * z**2)  

f_b = max(2, HIT_SD*6)
feat = np.exp(-.5 * (np.arange(-f_b, f_b+1)/HIT_SD)**2)

def _decode(sig, peak_h, hit_offset):
    corr = scipy.signal.correlate(sig, feat, mode='same')
    hit_peaks = scipy.signal.find_peaks(corr, height=peak_h)[0] + hit_offset
    return hit_peaks.astype(int).tolist()

def decode_hit(sig):
    # return _decode(sig, peak_h = .5, hit_offset = 0)
    sig_grad = np.gradient(sig)
    return sorted([
        *(scipy.signal.find_peaks(sig_grad)[0].astype(int)),
        *(scipy.signal.find_peaks(-sig_grad)[0].astype(int)),
    ])

def decode_hold(sig):
    sig_grad = np.gradient(sig)
    start_sig = np.maximum(0, sig_grad)
    end_sig = -np.minimum(0, sig_grad)

    start_idxs = _decode(start_sig, peak_h=.25, hit_offset=1)
    end_idxs = _decode(end_sig, peak_h=.25, hit_offset=-1)

    # ensure that first start is before first end
    while len(start_idxs) and len(end_idxs) and start_idxs[0] >= end_idxs[0]:
        end_idxs.pop(0)

    # ensure that there is one end for every start
    if len(start_idxs) > len(end_idxs):
        start_idxs = start_idxs[:len(end_idxs) - len(start_idxs)]
    elif len(end_idxs) > len(start_idxs):
        end_idxs = end_idxs[:len(start_idxs) - len(end_idxs)]
    
    return start_idxs, end_idxs