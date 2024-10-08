
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

import scipy.signal as signal

class AAUpsample1d(nn.Module):
    """up samples a signal via zero-padding + vanilla convolution + anti-aliasing + projection"""

    def __init__(self, dim: int, scale: int, kernel_size: int = 17):
        super().__init__()
        assert kernel_size % 2 == 1
        self.dim = dim
        self.scale = scale
        self.conv = nn.Conv1d(dim, dim, scale*2-1, 1, scale-1)
        self.proj_out = nn.Conv1d(dim, dim, 1)

        # sinc filter w/kaiser window
        beta = signal.kaiser_beta(signal.kaiser_atten(kernel_size, scale ** -1))
        kaiser = th.tensor(signal.windows.kaiser(kernel_size, beta))
        x = th.arange(kernel_size) - kernel_size // 2
        kernel = th.sinc(x/scale) * kaiser
        self.kernel = kernel / kernel.sum()

    def forward(self, x: Float[Tensor, "B D l"]) -> Float[Tensor, "B D L"]:
        b,d,l = x.size()
        upsampled = th.zeros(b,d,l*self.scale).to(x)
        upsampled[:,:,::self.scale] = x

        x = self.conv(upsampled)

        kernel = self.kernel[None,None,:].repeat(self.dim, 1, 1).to(x) # D 1 K
        filtered = F.conv1d(
            x, kernel,
            padding=kernel.size(-1) // 2,
            groups=self.dim,
        )

        return self.proj_out(filtered)