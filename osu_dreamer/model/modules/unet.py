
from jaxtyping import Float

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

        conv = lambda dim: nn.Sequential(
            nn.Conv1d(dim, dim, 5,1,2, groups=dim),
            nn.Conv1d(dim, dim, 1),
            nn.GroupNorm(1, dim),
            nn.SiLU(),
        )

        self.pre_split = nn.ModuleList()
        self.post_split = nn.ModuleList()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.pre_join = nn.ModuleList()
        self.post_join = nn.ModuleList()

        for scale in scales:
            self.chunk_size *= scale

            self.pre_split.append(ScaleShift(dim, t_dim, conv(dim)))
            self.post_split.append(ScaleShift(dim, t_dim, conv(dim)))
            self.down.append(Filter1D(dim, scale, transpose=False))

            self.up.insert(0, Filter1D(dim, scale, transpose=True))
            self.pre_join.insert(0, ScaleShift(dim, t_dim, conv(dim)))
            self.post_join.insert(0, ScaleShift(dim, t_dim, conv(dim)))

    def forward(
        self,
        x: Float[Tensor, "B D L"],
        t: Float[Tensor, "B T"],
    ) -> Float[Tensor, "B D L"]:
        
        x, p = pad(x, self.chunk_size)

        hs = []

        for pre_split, post_split, down in zip(self.pre_split, self.post_split, self.down):
            x = pre_split(x, t)
            hs.append(x)
            x = post_split(x, t)
            x = down(x)

        x = self.middle(x, t)

        for up, pre_join, post_join in zip(self.up, self.pre_join, self.post_join):
            x = up(x)
            x = pre_join(x, t)
            x = hs.pop() + x
            x = post_join(x, t)

        return unpad(x, p)