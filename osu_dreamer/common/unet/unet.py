
from typing import Optional
from collections.abc import Callable
from jaxtyping import Float

from torch import nn, Tensor

from osu_dreamer.common.residual import Residual
from osu_dreamer.common.filter import Filter1D

from .pad import pad, unpad
from .split_join import Split, Join

class UNetEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        scales: list[int],
        block: Callable[[], nn.Module],
    ):
        super().__init__()

        self.pre = nn.ModuleList()
        self.split = nn.ModuleList()
        self.down = nn.ModuleList()

        for scale in scales:
            self.pre.append(block())
            self.split.append(Split(dim))
            self.down.append(Filter1D(dim, scale, transpose=False))

    def forward(self, x: Float[Tensor, "B D L"]) -> tuple[list[Float[Tensor, "B D _L"]], Float[Tensor, "B D l"]]:
        hs = []
        for pre, split, down in zip(self.pre, self.split, self.down):
            h, x = split(pre(x))
            hs.append(h)
            x = down(x)
        return hs, x


class UNetDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        scales: list[int],
        block: Callable[[], nn.Module],
    ):
        super().__init__()

        self.post = nn.ModuleList()
        self.join = nn.ModuleList()
        self.up = nn.ModuleList()

        for scale in scales:
            self.post.insert(0, block())
            self.join.insert(0, Join(dim))
            self.up.insert(0, Filter1D(dim, scale, transpose=True))

    def forward(self, hs: list[Float[Tensor, "B D _L"]], x: Float[Tensor, "B D l"]) -> Float[Tensor, "B D L"]:
        for up, join, post in zip(self.up, self.join, self.post):
            x = up(x)
            h = hs.pop()
            x = post(join(h, x))
        return x

class UNet(nn.Module):
    def __init__(
        self,
        dim: int,
        scales: list[int],
        middle: nn.Module,
        block: Optional[Callable[[], nn.Module]] = None,
    ):
        super().__init__()

        if block is None:
            block = lambda: Residual(nn.Sequential(
                nn.Conv1d(dim, dim, 5,1,2, groups=dim),
                nn.Conv1d(dim, dim, 1),
                nn.GroupNorm(1, dim),
                nn.SiLU(),
                nn.Conv1d(dim, dim, 1),
            ))

        self.encoder = UNetEncoder(dim, scales, block)
        self.decoder = UNetDecoder(dim, scales, block)

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