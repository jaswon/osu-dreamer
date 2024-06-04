
from jaxtyping import Float

from torch import nn, Tensor
import torch.nn.functional as F

from .filter import Filter1D

def pad(x: Float[Tensor, "... L"], size: int) -> tuple[Float[Tensor, "... Lp"], int]:
    padding = (size-x.size(-1)%size)%size
    if padding > 0:
        x = F.pad(x, (0, padding))
    return x, padding

def unpad(x: Float[Tensor, "... Lp"], padding: int) -> Float[Tensor, "... L"]:
    if padding > 0:
        x = x[...,:-padding]
    return x

class Residual(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        
    def forward(self, x):
        return x + self.net(x)

class UNet(nn.Module):
    def __init__(
        self,
        dim: int,
        scales: list[int],
        middle: nn.Module,
        expand: int,
    ):
        super().__init__()

        self.middle = middle

        self.chunk_size = 1

        self.pre = nn.ModuleList()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.post = nn.ModuleList()

        block = lambda: Residual(nn.Sequential(
            Filter1D(dim, 1, transpose=False),
            nn.Conv1d(dim, expand*dim, 1),
            nn.GroupNorm(1, expand*dim),
            nn.SiLU(),
            nn.Conv1d(expand*dim, dim, 1),
        ))

        for scale in scales:
            self.chunk_size *= scale

            self.pre.append(block())
            self.down.append(Filter1D(dim, scale, transpose=False))

            self.up.insert(0, Filter1D(dim, scale, transpose=True))
            self.post.insert(0, block())

    def forward(
        self,
        x: Float[Tensor, "B D L"],
        *args, **kwargs,
    ) -> Float[Tensor, "B D L"]:
        
        x, p = pad(x, self.chunk_size)

        hs = []

        for pre, down in zip(self.pre, self.down):
            h = pre(x)
            hs.append(h)
            x = down(x-h)

        x = self.middle(x, *args, **kwargs)

        for up, post in zip(self.up, self.post):
            x = up(x)
            h = hs.pop()
            x = post(x+h)

        return unpad(x, p)