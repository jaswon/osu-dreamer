
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

    @cached_property
    def length(self) -> float:
        T,C = np.polynomial.legendre.leggauss(int(5*self.degree**.5))
        t = .5 * (T + 1) # map [-1,1] -> [0,1]
        f = np.linalg.norm(self.hodo().at(t), axis=0)
        return .5 * (C * f).sum()

    def hodo(self) -> "BezierCurve":
        return BezierCurve((self.degree-1) * (self.p[:,1:] - self.p[:,:-1]))

    def at(self, t: Float[ndarray, "T"]) -> Float[ndarray, "2 T"]:
        return (self._at_vs if self.degree > 12 else self._at_de_casteljau)(t)

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
            right.append(p[:,-1])
            if p.shape[1] == 1:
                break
            p = (1-t) * p[:,:-1] + t * p[:,1:]
        return BezierCurve(np.array(left).T), BezierCurve(np.array(right).T)