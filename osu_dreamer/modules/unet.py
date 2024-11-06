
from collections.abc import Callable
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

class UNetLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        scale: int,
        pre: nn.Module,
        middle: nn.Module,
    ):
        super().__init__()
        self.pre = pre
        self.skip = nn.Conv1d(dim, dim, scale*2-1, 1, scale-1)
        self.main = nn.Sequential(
            nn.Conv1d(dim, dim, scale, scale),
            middle,
            nn.ConvTranspose1d(dim, dim, scale, scale),
        )
        self.out = nn.Conv1d(2*dim, dim, 1)

    def forward(
        self,
        x: Float[Tensor, "B D L"],
    ) -> Float[Tensor, "B D L"]:
        x = self.pre(x)
        skip = self.skip(x)
        main = self.main(x)
        return self.out(th.cat([main, skip], dim=1))
        

class UNet(nn.Module):
    def __init__(
        self,
        dim: int,
        scales: list[int],
        block: Callable[[], nn.Module],
    ):
        super().__init__()

        net = block()
        self.chunk_size = 1
        for scale in scales:
            self.chunk_size *= scale
            net = UNetLayer(dim, scale, block(), net)

        self.net = net

    def forward(
        self,
        x: Float[Tensor, "B D L"],
    ) -> Float[Tensor, "B D L"]:
        padding = (self.chunk_size-x.size(-1)%self.chunk_size)%self.chunk_size
        if padding > 0:
            x = F.pad(x, (0, padding))

        x = self.net(x)

        if padding > 0:
            x = x[...,:-padding]
        return x