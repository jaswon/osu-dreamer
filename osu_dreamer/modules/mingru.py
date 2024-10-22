
from typing import Optional
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

def g(x):
    return th.where(x >= 0, x + 1, th.exp(x))

def log_g(x):
    return th.where(x >= 0, th.log(F.relu(x)+1), x)

class minGRU(nn.Module):
    """https://arxiv.org/pdf/2410.01201"""

    def __init__(self, dim: int, h_dim: int):
        super().__init__()
        self.hg = nn.Conv1d(dim, 2*h_dim, 1)

    def step(
        self,
        x: Float[Tensor, "B D 1"],
        prev_hidden: Optional[Float[Tensor, "B H 1"]] = None,
    ) -> Float[Tensor, "B H 1"]:
        h, gate = self.hg(x).chunk(2, dim=1)
        h = g(h)
        gate = gate.sigmoid()

        if prev_hidden is None:
            return h * gate
        return th.lerp(prev_hidden, h, gate)

    def forward(
        self,
        x: Float[Tensor, "B D L"],
        prev_hidden: Optional[Float[Tensor, "B H 1"]] = None,
    ) -> Float[Tensor, "B H L"]:
        h, gate = self.hg(x).chunk(2, dim=1)

        log_coeffs = -F.softplus(gate)
        log_values = -F.softplus(-gate) + log_g(h)

        if prev_hidden is not None:
            log_values = th.cat([prev_hidden.log(), log_values], dim=2)
            log_coeffs = F.pad(log_coeffs, (0,0, 0,0, 1,0))

        # heinsen associative scan (log-space)
        a_star = log_coeffs.cumsum(dim=2)
        log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=2)
        log_h = a_star + log_h0_plus_b_star
        h = log_h.exp()

        return h[:,:,-x.size(2):]

class minGRULayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        h_dim = dim // 2
        assert h_dim * 2 == dim

        self.fore = minGRU(dim, h_dim)
        self.back = minGRU(dim, h_dim)
        self.out = nn.Conv1d(dim, dim, 1)

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L"]:
        fore = self.fore(x)
        back = self.back(x.flip(2)).flip(2)
        return self.out(th.cat([fore, back], dim=1))