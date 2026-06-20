
from jaxtyping import Float

from torch import nn, Tensor

from einops.layers.torch import Rearrange

from osu_dreamer.modules.derf import Derf


class SpecFeatures(nn.Module):
    def __init__(
        self,
        n_freqs: int,
        d_a: int,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Unflatten(1, (1, -1)),
            nn.Conv2d(1, 8, (8,3), (6,1), (1,1), bias=False),
            Derf(8, 1, 1),
            nn.SiLU(),
            nn.Conv2d(8, 32, (6,3), (4,1), (1,1), bias=False),
            Derf(32, 1, 1),
            nn.SiLU(),
            Rearrange('b c a l -> b (c a) l'),
            nn.Conv1d(32*(n_freqs//24), d_a, 1, bias=False),
            Derf(d_a, 1),
            nn.SiLU(),
        )

    def forward(self, x: Float[Tensor, "B F L"]) -> Float[Tensor, "B A L"]:
        return self.net(x)