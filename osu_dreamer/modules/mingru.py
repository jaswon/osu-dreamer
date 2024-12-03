
from jaxtyping import Float, Complex

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

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

def min_gru_bidirectional(x: Float[Tensor, "B H ... L"]) -> Float[Tensor, "B H/2 ... L"]:
    fore, back = x.chunk(2, dim=1)
    return th.cat([
        min_gru(*fore.chunk(2, dim=1)),
        min_gru(*back.flip(2).chunk(2, dim=1)).flip(2),
    ], dim=1)

class minGRU2(nn.Module):
    def __init__(self): super().__init__()
    
    def forward(self, x: Float[Tensor, "B H ... L"]) -> Float[Tensor, "B H/2 ... L"]:
        return min_gru_bidirectional(x)