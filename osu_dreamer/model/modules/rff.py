
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

class RandomFourierFeatures(nn.Module):
    def __init__(
        self,
        in_dim: int, 
        n_feats: int,
        out_dim: int,
        scale: float = 10.,
    ):
        super().__init__()
        self.register_buffer('W', th.randn(in_dim, n_feats) * scale)
        self.out = nn.Linear(n_feats*2, out_dim)

    def forward(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... E"]:
        theta = (x * 2 * th.pi) @ self.W
        return self.out(th.cat([theta.sin(), theta.cos()], dim=-1))
    
class RandomFourierFeatures1d(nn.Module):
    def __init__(
        self,
        in_dim: int, 
        n_feats: int,
        out_dim: int,
        scale: float = 10.,
    ):
        super().__init__()
        self.register_buffer('W', th.randn(n_feats, in_dim, 1) * scale)
        self.out = nn.Conv1d(n_feats*2, out_dim, 1)

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B E L"]:
        theta = F.conv1d(x*2*th.pi, self.W)
        return self.out(th.cat([theta.sin(), theta.cos()], dim=1))