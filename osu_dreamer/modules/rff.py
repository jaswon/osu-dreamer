
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
    
class RandomFourierFeatures(nn.Module):
    def __init__(
        self,
        dim: int, 
        n_feats: int,
    ):
        super().__init__()
        self.register_buffer('f', th.randn(dim, n_feats))
        self.register_buffer('p', th.rand(n_feats))

    def forward(self, x: Float[Tensor, "B C"]) -> Float[Tensor, "B N"]:
        theta = 2 * th.pi * (x @ self.f + self.p)
        return theta.cos()