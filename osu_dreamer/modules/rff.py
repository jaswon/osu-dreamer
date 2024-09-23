
import torch as th
from torch import nn
import torch.nn.functional as F

from einops import rearrange

class RandomFourierFeatures(nn.Module):
    def __init__(
        self,
        dim: int, 
        n_feats: int,
        scale: float = 10.,
        conv1x1: bool = False,
    ):
        super().__init__()
        self.conv1x1 = conv1x1

        d = n_feats//2
        assert d * 2 == n_feats, '`n_feats` must be even'
        self.register_buffer('W', th.randn(dim, d) * scale)

    def forward(self, x):
        if self.conv1x1:
            theta = F.conv1d(x * 2 * th.pi, rearrange(self.W, 'i o -> o i 1'))
        else:
            theta = (x * 2 * th.pi) @ self.W
        return th.cat([theta.sin(), theta.cos()], dim=1 if self.conv1x1 else -1)