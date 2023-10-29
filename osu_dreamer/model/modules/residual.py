
from collections.abc import Sequence
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

class ResiDual(nn.Module):
    def __init__(
        self,
        dim: int,
        blocks: Sequence[nn.Module],
    ):
        super().__init__()

        self.blocks = nn.ModuleList(blocks)
        self.x_d_scale = 1 / len(blocks)

        self.norms = nn.ModuleList([
            nn.GroupNorm(1, dim)
            for _ in blocks
        ])
        self.post_norm = nn.GroupNorm(1, dim)
        

    def forward(
        self,
        x: Float[Tensor, "B D L"],
        *args,
        **kwargs,
    ) -> Float[Tensor, "B D L"]:
        # resiDual
        r = th.zeros_like(x)
        for block, norm in zip(self.blocks, self.norms):
            x_f = block(x, *args, **kwargs)
            x, r = norm(x + x_f), r + x_f * self.x_d_scale
        x = x + self.post_norm(r)
        return x