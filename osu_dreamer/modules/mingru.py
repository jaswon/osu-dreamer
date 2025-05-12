
from jaxtyping import Float, Complex

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

import osu_dreamer.modules.mp as MP

def complex_log(float_input: Float[Tensor, "..."], eps=1e-6) -> Complex[Tensor, "..."]:
    real = th.clamp_min(float_input.abs(), eps).log()
    imag = (float_input < 0) * th.pi
    return th.complex(real, imag.float())

def min_gru(
    h: Float[Tensor, "... L"], 
    g: Float[Tensor, "... L"],
) -> Float[Tensor, "... L"]:
    log_coeffs = -F.softplus(g)
    log_values = -F.softplus(-g) + complex_log(h)

    # heinsen associative scan (log-space)
    a_star = log_coeffs.cumsum(dim=-1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=-1)
    log_h = a_star + log_h0_plus_b_star
    
    return log_h.exp().real

class MinGRU(nn.Module):
    def __init__(self, dim: int, out_dim: int = None):
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