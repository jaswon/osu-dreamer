
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

log_g = lambda x: th.where(x >= 0, th.log(F.relu(x)+1), x)

def min_gru(hg: Float[Tensor, "B H L"]) -> Float[Tensor, "B H/2 L"]:
    h, gate = hg.chunk(2, dim=1)

    log_coeffs = -F.softplus(gate)
    log_values = -F.softplus(-gate) + log_g(h)

    # heinsen associative scan (log-space)
    a_star = log_coeffs.cumsum(dim=2)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=2)
    log_h = a_star + log_h0_plus_b_star
    
    return log_h.exp()

def min_gru_bidirectional(x: Float[Tensor, "B H L"]) -> Float[Tensor, "B H/2 L"]:
    fore, back = x.chunk(2, dim=1)
    return th.cat([ min_gru(fore), min_gru(back.flip(2)).flip(2) ], dim=1)

class minGRU2(nn.Module):
    def __init__(self): super().__init__()
    
    def forward(self, x: Float[Tensor, "B H L"]) -> Float[Tensor, "B H/2 L"]:
        return min_gru_bidirectional(x)