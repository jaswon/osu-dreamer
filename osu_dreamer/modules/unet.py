
from collections.abc import Callable
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

import scipy.signal as signal
from einops import repeat

class LPFUpsample(nn.Module):
    def __init__(
        self,
        dim: int, 
        scale: int, 
        kernel_size: int = 17,
    ):
        assert kernel_size % 2 == 1
        super().__init__()
        
        self.dim = dim
        self.pad = kernel_size // 2
        self.scale = scale

        # sinc filter w/kaiser window
        beta = signal.kaiser_beta(signal.kaiser_atten(kernel_size, scale ** -1))
        kaiser = th.tensor(signal.windows.kaiser(kernel_size, beta))
        X = th.arange(kernel_size) - kernel_size // 2
        K = th.sinc(X/scale) * kaiser
        K = K / K.sum()
        self.K = K

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L*{self.scale}"]:
        return F.conv1d(
            repeat(x, 'b d l -> b d (l s)', s=self.scale), 
            repeat(self.K.to(x), 'k -> d 1 k', d=self.dim),
            padding=self.pad,
            groups=self.dim,
        )

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
        self.down = nn.AvgPool1d(scale, scale)
        self.middle = middle
        self.up = LPFUpsample(dim, scale)
        self.gate = nn.Sequential(
            nn.Conv1d(2*dim, dim, 1),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),
        )
        self.out = nn.Conv1d(2*dim, dim, 1)

    def forward(
        self,
        x: Float[Tensor, "B D L"],
    ) -> Float[Tensor, "B D L"]:
        x = self.pre(x)
        skip = self.skip(x)
        down = self.down(x)
        middle = self.middle(down)
        up = self.up(middle)
        gate = self.gate(th.cat([up, skip], dim=1))
        mix = skip * gate
        return self.out(th.cat([up, mix], dim=1))
        

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