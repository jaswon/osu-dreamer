
from jaxtyping import Float

import torch as th
from torch import nn, Tensor


class FourierFeatures(nn.Module):
    def __init__(self, dim: int, features: int, freq_scale: float = 16.):
        super().__init__()
        self.proj_in = nn.Linear(dim, features)
        th.nn.init.normal_(self.proj_in.weight, std=freq_scale)
        th.nn.init.uniform_(self.proj_in.bias, -th.pi, th.pi)
        self.scale = (2/features) ** .5

    def forward(self, x: Float[Tensor, "*B I"]) -> Float[Tensor, "*B O"]:
        return self.scale * th.cos(self.proj_in(x))