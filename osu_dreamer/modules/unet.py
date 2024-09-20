
from collections.abc import Callable
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

import scipy.signal as signal
from einops import repeat

def pad(x: Float[Tensor, "... L"], size: int) -> tuple[Float[Tensor, "... Lp"], int]:
    padding = (size-x.size(-1)%size)%size
    if padding > 0:
        x = F.pad(x, (0, padding))
    return x, padding

def unpad(x: Float[Tensor, "... Lp"], padding: int) -> Float[Tensor, "... L"]:
    if padding > 0:
        x = x[...,:-padding]
    return x

def make_lpf(dim: int, scale: int, kernel_size: int = 17):
    assert kernel_size % 2 == 1

    # sinc filter w/kaiser window
    beta = signal.kaiser_beta(signal.kaiser_atten(kernel_size, scale ** -1))
    kaiser = th.tensor(signal.windows.kaiser(kernel_size, beta))
    X = th.arange(kernel_size) - kernel_size // 2
    K = th.sinc(X/scale) * kaiser
    K = K / K.sum()

    def lpf(x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L"]:
        return F.conv1d(
            x, repeat(K.to(x), 'k -> d 1 k', d=dim),
            padding=kernel_size // 2,
            groups=dim,
        )

    return lpf

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
        self.skip = nn.ModuleList()
        self.down = []
        self.up = []
        self.post = nn.ModuleList()
        
        def make_down_up(scale: int):
            lpf = make_lpf(dim, scale)
            return (
                lambda x: F.avg_pool1d(lpf(x), scale, scale),
                lambda x: lpf(repeat(x, 'b d l -> b d (l s)', s=scale)),
            )

        self.chunk_size = 1
        for scale in scales:
            self.chunk_size *= scale
            down, up = make_down_up(scale)

            self.pre.append(block())
            self.skip.append(nn.Conv1d(dim, dim, scale*2-1, 1, scale-1))
            self.down.append(down)

            self.up.insert(0, up)
            self.post.insert(0, block())

        self.middle = middle

    def forward(
        self,
        x: Float[Tensor, "B D L"],
        *args, **kwargs,
    ) -> Float[Tensor, "B D L"]:
        
        x, p = pad(x, self.chunk_size)

        hs = []
        for pre, skip, down in zip(self.pre, self.skip, self.down):
            x = pre(x, *args, **kwargs)
            hs.append(skip(x))
            x = down(x)
            
        x = self.middle(x, *args, **kwargs)

        for up, post in zip(self.up, self.post):
            x = up(x)
            x = hs.pop() + x
            x = post(x, *args, **kwargs)

        return unpad(x, p)