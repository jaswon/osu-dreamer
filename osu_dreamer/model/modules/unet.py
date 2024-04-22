
from collections.abc import Callable
from jaxtyping import Float

from torch import nn, Tensor
import torch.nn.functional as F

def get_padding(L: int, size: int) -> int:
    """returns padding to add to `L` to be a multiple of `size`"""
    return (size-L%size)%size

class UNet(nn.Module):
    def __init__(
        self,
        dim: int,
        scales: list[int],
        proj: Callable[[int], nn.Module],
        middle: Callable[[int], nn.Module],
    ):
        super().__init__()

        self.middle = middle(dim)

        self.chunk_size = 1

        self.pre = nn.ModuleList()
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.post = nn.ModuleList()

        for scale in scales:
            self.chunk_size *= scale

            self.pre.append(proj(dim))
            self.down.append(nn.Sequential(
                nn.Conv1d(dim, dim, scale, scale, groups=dim),
                nn.Conv1d(dim ,dim, 1),
                nn.GroupNorm(1, dim),
                nn.SiLU(),
            ))

            self.up.insert(0, nn.Sequential(
                nn.GroupNorm(1, dim),
                nn.SiLU(),
                nn.ConvTranspose1d(dim, dim, scale, scale, groups=dim),
                nn.Conv1d(dim ,dim, 1),
            ))
            self.post.insert(0, proj(dim))

    def forward(
        self,
        x: Float[Tensor, "B D L"],
        *args, **kwargs
    ) -> Float[Tensor, "B D L"]:
        
        pad = get_padding(x.size(-1), self.chunk_size)

        if pad > 0:
            x = F.pad(x, (0, pad), mode='replicate')

        hs = []

        for pre, down in zip(self.pre, self.down):
            x = pre(x, *args, **kwargs)
            hs.append(x)
            x = down(x)

        x = self.middle(x, *args, **kwargs)

        for up, post in zip(self.up, self.post):
            x = up(x)
            x = hs.pop() + x
            x = post(x, *args, **kwargs)

        if pad > 0:
            x = x[...,:-pad]

        return x