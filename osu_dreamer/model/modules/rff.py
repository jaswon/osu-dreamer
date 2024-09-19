
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

class RandomFourierFeatures(nn.Module):
    def __init__(
        self,
        dim: int, 
        n_feats: int,
        scale: float = 10.,
    ):
        super().__init__()
        d = n_feats//2
        assert d * 2 == n_feats, '`n_feats` must be even'
        self.register_buffer('W', th.randn(dim, d) * scale)

    def forward(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... E"]:
        theta = (x * 2 * th.pi) @ self.W
        return th.cat([theta.sin(), theta.cos()], dim=-1)
    
class  RandomFourierFeatures1d(nn.Module):
    def __init__(
        self,
        dim: int, 
        n_feats: int,
        scale: float = 10.,
    ):
        super().__init__()
        d = n_feats//2
        assert d * 2 == n_feats, '`n_feats` must be even'
        self.register_buffer('W', th.randn(d, dim, 1) * scale)

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B E L"]:
        theta = F.conv1d(x * 2 * th.pi, self.W)
        return th.cat([theta.sin(), theta.cos()], dim=1)