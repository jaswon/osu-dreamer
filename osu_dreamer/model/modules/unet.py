
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from .filter import Filter1D
from .scaleshift import ScaleShift

def pad(x: Float[Tensor, "... L"], size: int) -> tuple[Float[Tensor, "... Lp"], int]:
    padding = (size-x.size(-1)%size)%size
    if padding > 0:
        x = F.pad(x, (0, padding))
    return x, padding

def unpad(x: Float[Tensor, "... Lp"], padding: int) -> Float[Tensor, "... L"]:
    if padding > 0:
        x = x[...,:-padding]
    return x


class UNet(nn.Module):
    def __init__(
        self,
        dim: int,
        t_dim: int,
        scales: list[int],
        middle: nn.Module,
    ):
        super().__init__()

        self.middle = middle

        self.chunk_size = 1

        self.split = nn.ModuleList()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.join = nn.ModuleList()

        for scale in scales:
            self.chunk_size *= scale

            self.split.append(ScaleShift(dim, t_dim, nn.Sequential(
                nn.Conv1d(dim, dim, 5,1,2, groups=dim),
                nn.Conv1d(dim, 2*dim, 1),
                nn.GroupNorm(1, 2*dim),
                nn.SiLU(),
                nn.Conv1d(2*dim, 2*dim, 1),
            )))
            self.down.append(Filter1D(dim, scale, transpose=False))

            self.up.insert(0, Filter1D(dim, scale, transpose=True))
            self.join.insert(0, ScaleShift(2*dim, t_dim, nn.Sequential(
                nn.Conv1d(2*dim, 2*dim, 5,1,2, groups=2*dim),
                nn.Conv1d(2*dim, dim, 1),
                nn.GroupNorm(1, dim),
                nn.SiLU(),
                nn.Conv1d(dim, dim, 1),
            )))

    def forward(
        self,
        x: Float[Tensor, "B D L"],
        t: Float[Tensor, "B T"],
    ) -> Float[Tensor, "B D L"]:
        
        x, p = pad(x, self.chunk_size)

        hs = []

        for split, down in zip(self.split, self.down):
            x, h = split(x, t).chunk(2, dim=1)
            hs.append(h)
            x = down(x)

        x = self.middle(x, t)

        for up, join in zip(self.up, self.join):
            x = up(x)
            h = hs.pop()
            x = join(th.cat([h, x], dim=1), t)

        return unpad(x, p)