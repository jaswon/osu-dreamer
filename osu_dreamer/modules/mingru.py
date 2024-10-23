
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from osu_dreamer.modules.modconv import ModulatedConv1d

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

class minGRULayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        h_dim = dim // 2
        assert h_dim * 2 == dim

        self.fore = nn.Conv1d(dim, dim, 1)
        self.back = nn.Conv1d(dim, dim, 1)

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L"]:
        return th.cat([
            min_gru(self.fore(x)),
            min_gru(self.back(x.flip(2))).flip(2),
        ], dim=1)

class minGRUModLayer(nn.Module):
    def __init__(self, dim: int, c_dim: int):
        super().__init__()
        h_dim = dim // 2
        assert h_dim * 2 == dim

        self.fore = ModulatedConv1d(dim, dim, c_dim)
        self.back = ModulatedConv1d(dim, dim, c_dim)

    def forward(self, x: Float[Tensor, "B D L"], c: Float[Tensor, "B C"]) -> Float[Tensor, "B D L"]:
        return th.cat([
            min_gru(self.fore((x,c))),
            min_gru(self.back((x.flip(2),c))).flip(2),
        ], dim=1)