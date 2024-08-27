
from collections.abc import Sequence
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(th.ones(dim,1))

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L"]:
        return F.normalize(x, dim = 1, eps=1e-8) * self.gamma * self.scale

class ResStack(nn.Module):
    def __init__(
        self,
        dim: int,
        blocks: Sequence[nn.Module],
    ):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.outs = nn.ModuleList([ nn.Conv1d(dim, 2*dim, 1) for _ in blocks ])
        self.norms = nn.ModuleList([ RMSNorm(dim) for _ in blocks ])

    def forward(
        self,
        x: Float[Tensor, "B D L"],
        *args, **kwargs,
    ) -> Float[Tensor, "B D L"]:
        
        o = x
        for block, out, norm in zip(self.blocks, self.outs, self.norms):
            h = block(x, *args, **kwargs)
            res, skip = out(h).chunk(2, dim=1)
            x = F.silu(norm(x + res))
            o = o + skip

        return o