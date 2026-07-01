
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

class RMSNorm1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(th.randn(dim, 1))

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L"]:
        xf = x.float()
        inv_rms = xf.pow(2).mean(dim=1, keepdim=True).add(1e-6).rsqrt()
        return (xf * inv_rms).to(x.dtype) * self.gamma