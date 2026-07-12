
from jaxtyping import Float

import torch as th
from torch import nn, Tensor


class FourierFeatures(nn.Module):
    def __init__(self, dim: int, features: int, n_bins: int = 16):
        super().__init__()
        freq_scale = float(n_bins)
        self.W: th.Tensor; self.register_buffer('W', th.randn(features, dim) * freq_scale)
        self.b: th.Tensor; self.register_buffer('b', th.empty(features).uniform_(-th.pi, th.pi))
        self.scale = (2/features) ** .5

    def forward(self, x: Float[Tensor, "*B I"]) -> Float[Tensor, "*B O"]:
        return self.scale * th.cos(x @ self.W.T + self.b)