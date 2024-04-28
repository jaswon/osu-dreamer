
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

        normact = lambda: nn.Sequential(
            nn.GroupNorm(1, dim),
            nn.SiLU(),
        )

        self.normacts = nn.ModuleList([ normact() for _ in blocks ])
        self.post_normact = normact()
        

    def forward(
        self,
        x: Float[Tensor, "B D L"],
        *args,
        **kwargs,
    ) -> Float[Tensor, "B D L"]:
        r = th.zeros_like(x)
        for block, norm in zip(self.blocks, self.normacts):
            x_f = block(x, *args, **kwargs)
            x, r = norm(x + x_f), r + x_f * self.x_d_scale
        x = x + self.post_normact(r)
        return x