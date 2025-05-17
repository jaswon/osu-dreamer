
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from einops import rearrange, repeat


def rotate_half(x: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:
    x1, x2 = rearrange(x, '... (d r) -> ... d r', r = 2).unbind(dim = -1)
    x_r = th.stack((-x2, x1), dim = -1)
    return rearrange(x_r, '... d r -> ... (d r)')

class RoPE(nn.Module):
    def __init__(
        self, 
        dim: int, 
        max_timescale: float = 10_000,
    ):
        """
        - `max_timescale` should be at least 2x the largest difference expected between any two positions
        """
        super().__init__()
        d = dim // 2
        assert d * 2 == dim
        self.dim = dim
        self.fs = 2 * th.pi * max_timescale ** th.linspace(0, -1, d) # D/2

    def forward(
        self, 
        x: Float[Tensor, "B N D"],
        t: Float[Tensor, "B N"]
    ) -> Float[Tensor, "B N D"]:
        theta = t[:,:,None] * self.fs # B N D/2
        theta = repeat(theta, 'l d -> l (d r)', r=2) # B N D
        return x * theta.cos() + rotate_half(x) * theta.sin()