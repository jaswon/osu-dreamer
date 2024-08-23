
from collections.abc import Callable
from jaxtyping import Float

from torch import nn, Tensor

from .pad import pad, unpad

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
            self.split.append(nn.Conv1d(dim, 2*dim, 1))
            self.down.append(nn.Conv1d(dim, dim, scale, scale, groups=dim))
            
            self.post.insert(0, block())
            self.up.insert(0, nn.ConvTranspose1d(dim, dim, scale, scale, groups=dim))

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
            h, x = split(x).chunk(2, dim=1)
            hs.append(h)
            x = down(x)
            
        x = self.middle(x, *args, **kwargs)

        for up, post in zip(self.up, self.post):
            x = up(x)
            x = hs.pop() * x
            x = post(x, *args, **kwargs)

        return unpad(x, p)