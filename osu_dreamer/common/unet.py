
from jaxtyping import Float

from torch import nn, Tensor
import torch.nn.functional as F

from .filter import Filter1D
from .residual import ResStack

def pad(x: Float[Tensor, "... L"], size: int) -> tuple[Float[Tensor, "... Lp"], int]:
    padding = (size-x.size(-1)%size)%size
    if padding > 0:
        x = F.pad(x, (0, padding))
    return x, padding

def unpad(x: Float[Tensor, "... Lp"], padding: int) -> Float[Tensor, "... L"]:
    if padding > 0:
        x = x[...,:-padding]
    return x

block = lambda dim, block_depth: ResStack(dim, [
    nn.Sequential(
        Filter1D(dim, 1, transpose=False),
        nn.Conv1d(dim, dim, 1),
        nn.GroupNorm(1, dim),
        nn.SiLU(),
    )
    for _ in range(block_depth)
])

class UNetEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        scales: list[int],
        block_depth: int,
    ):
        super().__init__()

        self.pre = nn.ModuleList()
        self.down = nn.ModuleList()

        for scale in scales:
            self.pre.append(block(dim, block_depth))
            self.down.append(Filter1D(dim, scale, transpose=False))

    def forward(self, x: Float[Tensor, "B X L"]) -> tuple[list[Float[Tensor, "B X _L"]], Float[Tensor, "B X l"]]:
        hs = []
        for pre, down in zip(self.pre, self.down):
            h = pre(x)
            hs.append(h)
            x = down(x-h)
        return hs, x


class UNetDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        scales: list[int],
        block_depth: int,
    ):
        super().__init__()

        self.post = nn.ModuleList()
        self.up = nn.ModuleList()

        for scale in scales:
            self.post.insert(0, block(dim, block_depth))
            self.up.insert(0, Filter1D(dim, scale, transpose=True))

    def forward(self, hs: list[Float[Tensor, "B X _L"]], x: Float[Tensor, "B X l"]) -> Float[Tensor, "B X L"]:
        for up, post in zip(self.up, self.post):
            x = up(x)
            h = hs.pop()
            x = post(x+h)
        return x

class UNet(nn.Module):
    def __init__(
        self,
        dim: int,
        scales: list[int],
        block_depth: int,
        middle: nn.Module,
    ):
        super().__init__()

        self.encoder = UNetEncoder(dim, scales, block_depth)
        self.decoder = UNetDecoder(dim, scales, block_depth)

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

        hs, x = self.encoder(x)
        x = self.middle(x, *args, **kwargs)
        x = self.decoder(hs, x)

        return unpad(x, p)