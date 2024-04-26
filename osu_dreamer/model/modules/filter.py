
from functools import partial
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F


class Filter1D(nn.Module):
    def __init__(
        self,
        channels: int,
        stride: int,
        transpose: bool,
        filter_size: int = 15,
    ):
        super().__init__()
        assert filter_size % 2 == 1

        # compute sinc filter
        x = th.arange(filter_size) - filter_size // 2
        filter = th.sinc(x / stride) / stride
        filter = filter[None,None,:].repeat(channels, 1, 1) # D 1 K
        self.filter = nn.Parameter(filter)
        setattr(self.filter, 'opt_adj', 'filter')

        self.padding = filter_size // 2
        self.channels = channels
        self.stride = stride
        self.transpose = transpose

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D l"]:
        if self.transpose:
            conv = partial(F.conv_transpose1d, output_padding=self.stride-1)
        else:
            conv = F.conv1d

        return conv(
            x, self.filter.to(x),
            stride=self.stride,
            padding=self.padding,
            groups=self.channels,
        )