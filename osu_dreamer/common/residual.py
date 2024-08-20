
from collections.abc import Sequence
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from osu_dreamer.common.norm import RMSNorm

class Residual(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return x + self.net(x)

class ResStack(nn.Module):
    def __init__(
        self,
        dim: int,
        blocks: Sequence[nn.Module],
    ):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.norms = nn.ModuleList([ RMSNorm(dim) for _ in blocks ])
        self.outs = nn.ModuleList([ nn.Conv1d(dim, 2*dim, 1) for _ in blocks ])

    def forward(
        self,
        x: Float[Tensor, "B D L"],
        *args, **kwargs,
    ) -> Float[Tensor, "B D L"]:
        
        o = th.zeros_like(x)
        for block, norm, out in zip(self.blocks, self.norms, self.outs):
            h = block(x, *args, **kwargs)
            res, skip = out(h).chunk(2, dim=1)
            x = F.silu(norm(x + res))
            o = o + skip

        return o
