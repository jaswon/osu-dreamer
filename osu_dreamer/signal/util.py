import numpy as np

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