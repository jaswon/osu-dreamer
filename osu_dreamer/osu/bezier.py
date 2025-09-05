
from jaxtyping import Float

from functools import cached_property

from numpy import ndarray
import numpy as np

class BezierCurve:
    def __init__(self, p: Float[ndarray, "2 N"]):
        assert p.shape[1] > 0
        self.p = p
    
    def __repr__(self):
        return f"{self.__class__.__name__}({repr(self.p)})"

    @property
    def degree(self):
        return self.p.shape[1]

    def _length_at(self, t: float) -> float:
        assert 0 <= t <= 1
        if t == 0: return 0.
            
        T, C = np.polynomial.legendre.leggauss(int(5 * self.degree ** .5))
        
        # map [-1,1] -> [0, t]
        t_samples = t/2 * (T + 1)
        
        # speed at sample points
        speeds = np.linalg.norm(self.hodo().at(t_samples), axis=0)
        
        return t/2 * (C * speeds).sum()

    @cached_property
    def length(self) -> float:
        return self._length_at(1.)

    def hodo(self) -> "BezierCurve":
        return BezierCurve((self.degree-1) * (self.p[:,1:] - self.p[:,:-1]))

    def at(self, t: Float[ndarray, "T"]) -> Float[ndarray, "2 T"]:
        return self._at_de_casteljau(t)

    def _at_vs(self, t: Float[ndarray, "T"]) -> Float[ndarray, "2 T"]:

        # flip t to avoid divide by zero
        flip_t = t > .5
        t = np.where(flip_t, 1-t, t)
        p = np.where(flip_t[None,None], np.flip(self.p[:,:,None], axis=1), self.p[:,:,None])

        j = np.arange(self.degree)[:,None]
        nCj_factor = (self.degree-j) / ((j+self.degree-1) % self.degree + 1)
        b = np.cumprod(nCj_factor * t, axis=0) * p / np.where(t==0,1,t)

        a = np.full((self.degree, len(t)), 1-t)[None] # D N T
        a_star = np.cumprod(a, axis=1)
        b_star = np.cumsum(b / a_star, axis=1)
        x = a_star * b_star
        
        return x[:,-1]
    
    def _at_de_casteljau(self, t: Float[ndarray, "T"]) -> Float[ndarray, "2 T"]:
        p = np.repeat(self.p[:,:,None], len(t), -1) # D N T
        while p.shape[1] > 1:
            p = (1-t) * p[:,:-1] + t * p[:,1:]
        return p[:,0]
    
    def split_at(self, t: float) -> tuple["BezierCurve", "BezierCurve"]:
        assert 0 <= t <= 1
        p, left, right = self.p, [], []
        while True:
            left.append(p[:,0])
            right.insert(0, p[:,-1])
            if p.shape[1] == 1:
                break
            p = (1-t) * p[:,:-1] + t * p[:,1:]
        return BezierCurve(np.array(left).T), BezierCurve(np.array(right).T)

    def _find_t_for_length(self, target_length: float) -> float:
        if target_length <= 0:
            return 0.0
        if target_length >= self.length:
            return 1.0

        # Binary search for t
        low_t, high_t = 0.0, 1.0
        
        for _ in range(20): # 20 iterations give plenty of precision
            mid_t = (low_t + high_t) / 2
            current_length = self._length_at(mid_t)
            
            if current_length < target_length:
                low_t = mid_t
            else:
                high_t = mid_t
                
        return (low_t + high_t) / 2

    def split_at_length(self, fraction: float) -> tuple["BezierCurve", "BezierCurve"]:
        """
        Splits the curve at a point specified by a fraction of its total arc length.
        """
        assert 0 <= fraction <= 1
        
        target_length = fraction * self.length
        t = self._find_t_for_length(target_length)
        return self.split_at(t)