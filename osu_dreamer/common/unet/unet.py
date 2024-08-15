
from collections.abc import Callable
from jaxtyping import Float

from torch import nn, Tensor

from osu_dreamer.common.norm import RMSNorm

from .pad import pad, unpad

class Join(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, 5,1,2, groups=dim),
            nn.Conv1d(dim, dim, 1),
            RMSNorm(dim),
            nn.SiLU(),
        )

    def forward(self, h: Float[Tensor, "B X L"], x: Float[Tensor, "B X L"]) -> Float[Tensor, "B X L"]:
        return self.net(h) * x

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
        self.down = nn.ModuleList()
        self.post = nn.ModuleList()
        self.join = nn.ModuleList()
        self.up = nn.ModuleList()

        for scale in scales:
            self.pre.append(block())
            self.down.append(nn.Conv1d(dim, dim, scale+2,scale,1))
            
            self.post.insert(0, block())
            self.join.insert(0, Join(dim))
            self.up.insert(0, nn.ConvTranspose1d(dim, dim, scale+2,scale,1))

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
        for pre, down in zip(self.pre, self.down):
            x = pre(x)
            hs.append(x)
            x = down(x)
            
        x = self.middle(x, *args, **kwargs)

        for up, join, post in zip(self.up, self.join, self.post):
            x = up(x)
            x = join(hs.pop(), x)
            x = post(x)

        return unpad(x, p)