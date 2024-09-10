
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from einops import repeat

import scipy.signal as signal

class LowPassFilter1D(nn.Module):
    def __init__(self, dim: int, scale: int, kernel_size: int = 17):
        super().__init__()
        assert kernel_size % 2 == 1
        self.dim = dim

        # sinc filter w/kaiser window
        beta = signal.kaiser_beta(signal.kaiser_atten(kernel_size, scale ** -1))
        kaiser = th.tensor(signal.windows.kaiser(kernel_size, beta))
        x = th.arange(kernel_size) - kernel_size // 2
        kernel = th.sinc(x/scale) * kaiser
        self.kernel = kernel / kernel.sum()

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L"]:
        kernel = repeat(self.kernel.to(x), 'k -> d 1 k', d=self.dim)
        return F.conv1d(
            x, kernel,
            padding=kernel.size(-1) // 2,
            groups=self.dim,
        )
    
class AADownsample1D(nn.Module):
    def __init__(self, dim: int, scale: int, kernel_size: int = 17):
        super().__init__()
        self.scale = scale
        self.filter = LowPassFilter1D(dim, scale, kernel_size)
        self.conv = nn.Conv1d(dim, dim, scale*2-1, 1, scale-1)

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D l"]:
        x = self.conv(x)
        x = self.filter(x)
        return F.avg_pool1d(x, self.scale, self.scale)

class AAUpsample1D(nn.Module):
    def __init__(self, dim: int, scale: int, kernel_size: int = 17):
        super().__init__()
        self.scale = scale
        self.filter = LowPassFilter1D(dim, scale, kernel_size)
        self.conv = nn.Conv1d(dim, dim, scale*2-1, 1, scale-1)

    def forward(self, x: Float[Tensor, "B D l"]) -> Float[Tensor, "B D L"]:
        x = repeat(x, 'b d l -> b d (l s)', s=self.scale)
        x = self.filter(x)
        return self.conv(x)