
from collections.abc import Callable
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

import scipy.signal as signal
from einops import repeat

from osu_dreamer.modules.film import FiLM

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

class UNetLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        t_dim: int,
        scale: int,
        pre: nn.Module,
        middle: nn.Module,
    ):
        super().__init__()

        self.pre = pre
            
        self.norm1 = FiLM(dim, t_dim)
        self.skip = nn.Conv1d(dim, dim, scale*2-1, 1, scale-1)

        self.down = nn.AvgPool1d(scale, scale)

        self.norm2 = FiLM(dim, t_dim)
        self.middle = middle
        
        lpf = make_lpf(dim, scale)
        self.up = lambda x: lpf(repeat(x, 'b d l -> b d (l s)', s=scale))
        
        self.norm3 = FiLM(2*dim, t_dim)
        self.gate = nn.Sequential(
            nn.Conv1d(2*dim, dim, 1),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),
        )
        
        self.norm4 = FiLM(2*dim, t_dim)
        self.out = nn.Conv1d(2*dim, dim, 1)

    def forward(
        self,
        x: Float[Tensor, "B D L"],
        t: Float[Tensor, "B T"],
    ) -> Float[Tensor, "B D L"]:
        x = self.pre(x, t)
        skip = self.skip(self.norm1(x, t))
        down = self.down(x)
        middle = self.middle(self.norm2(down, t), t)
        up = self.up(middle)
        gate = self.gate(self.norm3(th.cat([up, skip], dim=1), t))
        mix = skip * gate
        return self.out(self.norm4(th.cat([up, mix], dim=1), t))
        

class UNet(nn.Module):
    def __init__(
        self,
        dim: int,
        t_dim: int,
        scales: list[int],
        middle: nn.Module,
        block: Callable[[], nn.Module],
    ):
        super().__init__()

        net = middle
        self.chunk_size = 1
        for scale in scales:
            self.chunk_size *= scale
            net = UNetLayer(dim, t_dim, scale, block(), net)

        self.net = net

    def forward(
        self,
        x: Float[Tensor, "B D L"],
        t: Float[Tensor, "B T"],
    ) -> Float[Tensor, "B D L"]:
        padding = (self.chunk_size-x.size(-1)%self.chunk_size)%self.chunk_size
        if padding > 0:
            x = F.pad(x, (0, padding))

        x = self.net(x, t)

        if padding > 0:
            x = x[...,:-padding]
        return x