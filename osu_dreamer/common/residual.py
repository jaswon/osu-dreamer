
from collections.abc import Sequence
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

class ResStack(nn.Module):
    def __init__(
        self,
        dim: int,
        blocks: Sequence[nn.Module],
    ):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.outs = nn.ModuleList([ nn.Conv1d(dim, 2*dim, 1) for _ in blocks ])

    def forward(
        self,
        x: Float[Tensor, "B D L"],
        *args, **kwargs,
    ) -> Float[Tensor, "B D L"]:
        
        o = th.zeros_like(x)
        for block, out in zip(self.blocks, self.outs):
            h = block(x, *args, **kwargs)
            res, skip = out(h).chunk(2, dim=1)
            x = x + res
            o = o + skip

        return o
