
from jaxtyping import Float

from torch import nn, Tensor

from einops.layers.torch import Rearrange

import osu_dreamer.modules.mp as MP

class SqueezeExcitation(nn.Module):
    def __init__(self, dim: int, squeeze: int):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange('b c 1 -> b c'),
            MP.Linear(dim, dim // squeeze),
            MP.SiLU(),
            MP.Linear(dim // squeeze, dim),
            MP.Sigmoid(),
            Rearrange('b c -> b c 1'),
        )

    def forward(self, x: Float[Tensor, "B X L"]) -> Float[Tensor, "B X L"]:
        return x * self.se(x)