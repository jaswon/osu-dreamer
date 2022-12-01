import numpy as np
import scipy

# std dev of impulse indicating a hit
HIT_SD = 5

def smooth_hit(x: np.ndarray, mu: "Union[float, Tuple[float, float]]", sigma: float = HIT_SD):
    """
    a smoothed impulse
    modelled using a normal distribution with mean `mu` and std. dev `sigma`
    evaluated at values in `x`
    """
    
    if isinstance(mu, (float, int)):
        z = (x-mu)/sigma
    elif isinstance(mu, tuple):
        a,b = mu
        z = np.where(x<a,x-a,np.where(x<b,0,x-b)) / sigma
    else:
        raise NotImplementedError(f"`mu` must be float or tuple, got {type(mu)}")
        
    return np.exp(-.5 * z**2)      

    
f_b = max(2, HIT_SD*6)
feat = smooth_hit(np.arange(-f_b, f_b+1), 0)    

def _decode(sig, peak_h, hit_offset):
    corr = scipy.signal.correlate(sig, feat, mode='same')
    hit_peaks = scipy.signal.find_peaks(corr, height=peak_h)[0] + hit_offset
    return hit_peaks.astype(int).tolist()
    

def decode_hit(sig):
    return _decode(sig, peak_h = .5, hit_offset = 0)

def decode_hold(sig):
    sig_grad = np.gradient(sig)
    start_sig = np.maximum(0, sig_grad)
    end_sig = -np.minimum(0, sig_grad)
    
    return (
        _decode(start_sig, peak_h=.25, hit_offset=1),
        _decode(end_sig, peak_h=.25, hit_offset=-1),
    )