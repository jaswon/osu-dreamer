
from collections.abc import Callable
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from .filter import AAUpsample1d

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
        scales: list[int],
        middle: nn.Module,
        block: Callable[[], nn.Module],
    ):
        super().__init__()

        self.pre = nn.ModuleList()
        self.split = nn.ModuleList()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.post = nn.ModuleList()

        for scale in scales:
            self.pre.append(block())
            self.split.append(nn.Conv1d(dim, dim, scale*2-1, 1, scale-1))
            self.down.append(nn.Conv1d(dim, dim, scale, scale))
            
            self.up.insert(0, AAUpsample1d(dim, scale))
            self.post.insert(0, block())

        self.middle = middle

        self.chunk_size = 1
        for scale in scales:
            self.chunk_size *= scale

    def forward(
        self,
        x: Float[Tensor, "B D L"],
        *args, **kwargs,
    ) -> Float[Tensor, "B D L"]:
        
        x, p = pad(x, self.chunk_size)

        hs = []
        for pre, split, down in zip(self.pre, self.split, self.down):
            x = pre(x, *args, **kwargs)
            hs.append(split(x))
            x = down(x)
            
        x = self.middle(x, *args, **kwargs)

        for up, post in zip(self.up, self.post):
            x = up(x)
            x = hs.pop() * x
            x = post(x, *args, **kwargs)

        return unpad(x, p)