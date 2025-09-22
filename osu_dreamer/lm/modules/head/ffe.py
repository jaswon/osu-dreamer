
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
    
class FourierFeatureEmbedding(nn.Module):
    def __init__(self, dim: int, n_freqs: int, sigma: float = 10.):
        super().__init__()
        self.ffe_freqs: th.Tensor
        self.register_buffer('ffe_freqs', 2 * th.pi * sigma * th.randn(dim, n_freqs))

    def forward(self, x: Float[Tensor, "*B D"]) -> Float[Tensor, "*B F"]:
        assert (x >= 0).all()
        assert (x <= 1).all()
        thetas = x @ self.ffe_freqs.to(x.device)  # [*B, n_freqs]
        return th.cat([th.sin(thetas), th.cos(thetas)], dim=-1)