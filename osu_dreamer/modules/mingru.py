
from jaxtyping import Float, Complex, Inexact

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

import osu_dreamer.modules.mp as MP

def complex_log(float_input: Float[Tensor, "..."], eps=1e-6) -> Complex[Tensor, "..."]:
    real = th.clamp_min(float_input.abs(), eps).log()
    imag = (float_input < 0) * th.pi
    return th.complex(real, imag.to(real.dtype))

def log_parallel_scan(
    log_a: Inexact[Tensor, "... L"],   # log(a[0:t])
    log_b: Inexact[Tensor, "... L"],   # log(b[0:t])
    log_h0: Inexact[Tensor, "..."],
) -> Float[Tensor, "... L"]:
    """
    heinsen associative scan (log-space)
    
    returns h[0:t] such that:
        - h[-1] = h0
        - h[t] = a[t] * h[t-1] + b[t]
    """
    log_b = th.cat([log_h0[...,None], log_b], dim=-1)
    a_star = F.pad(log_a.cumsum(dim=-1), (1,0))
    log_h0_plus_b_star = th.logcumsumexp(log_b - a_star, dim=-1)
    log_h = a_star + log_h0_plus_b_star
    return log_h[...,1:].exp().real

def min_gru(
    h: Float[Tensor, "... L"], 
    g: Float[Tensor, "... L"],
    h0: None | Float[Tensor, "..."] = None,
) -> Float[Tensor, "... L"]:
    """ o[t] = (1-σ(g[t])) * o[t-1] + σ(g[t]) * h[t] """
    if h0 is None:
        h0 = th.zeros_like(h[...,0])
    return log_parallel_scan(
        F.logsigmoid(-g),                   # 1-σ(g)
        F.logsigmoid(g) + complex_log(h),   # σ(g) * h,
        h0.log(),
    ).to(h.dtype)

def min_gru_recurrent(
    h: Float[Tensor, "..."],
    g: Float[Tensor, "..."],
    o: Float[Tensor, "..."],
) -> Float[Tensor, "..."]:
    """ o[t] = (1-σ(g[t])) * o[t-1] + σ(g[t]) * h[t] """
    return th.lerp(o.to(h.dtype), h, F.sigmoid(g))

class MinGRU(nn.Module):
    def __init__(self, dim: int, out_dim: int | None = None):
        super().__init__()
        out_dim = out_dim or dim
        self.conv = MP.Conv1d(dim, dim, 5,1,2, groups=dim)
        self.h = MP.Conv1d(dim, out_dim, 1)
        self.g = MP.Conv1d(dim, out_dim-2, 1)

    def _polarize(self, g: Float[Tensor, "B D L"]) -> Float[Tensor, "B D+2 L"]:
        """https://arxiv.org/abs/2501.00658"""

        *b, _, l = g.shape
        g0 = g.new_full([*b, 1, l], -1000)
        g1 = g.new_full([*b, 1, l], +1000)
        return th.cat([ g0, g1, g ], dim=-2)

    def forward( self, x: Float[Tensor, "B D L"] ) -> Float[Tensor, "B O L"]:
        c = self.conv(x)
        return min_gru(self.h(c), self._polarize(self.g(c)))

def min_gru_bidirectional(x: Float[Tensor, "B H ... L"]) -> Float[Tensor, "B H/2 ... L"]:
    fore, back = x.chunk(2, dim=1)
    return th.cat([
        min_gru(*fore.chunk(2, dim=1)),
        min_gru(*back.flip(2).chunk(2, dim=1)).flip(2),
    ], dim=1)

class MinGRU2(nn.Module):
    def __init__(self): super().__init__()
    
    def forward(self, x: Float[Tensor, "B H ... L"]) -> Float[Tensor, "B H/2 ... L"]:
        return min_gru_bidirectional(x)