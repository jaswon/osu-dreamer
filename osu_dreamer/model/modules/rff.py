
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
    
class RandomFourierFeatures(nn.Module):
    def __init__(
        self,
        in_dim: int, 
        n_feats: int,
        scale: float = 10.,
    ):
        super().__init__()
        self.W = nn.Parameter(
            th.randn(in_dim, n_feats) * scale, 
            requires_grad=False,
        )

    def forward(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... E"]:
        theta = (x * 2 * th.pi) @ self.W
        return th.cat([theta.sin(), theta.cos()], dim=-1)