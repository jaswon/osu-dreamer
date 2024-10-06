
from typing import Callable
from jaxtyping import Float

from dataclasses import dataclass

import torch as th
from torch import nn, Tensor

@dataclass
class WaveNetArgs:
    num_stacks: int
    stack_depth: int

class WaveNet(nn.Module):
    """wavenet receptive field: 1+s*(2**d-1))"""
    def __init__(
        self,
        dim: int,
        args: WaveNetArgs,
        block: Callable[[int], nn.Module],
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.proj_in = nn.ModuleList()
        self.proj_out = nn.ModuleList()
        for _ in range(args.num_stacks):
            for d in range(args.stack_depth):
                self.proj_in.append(nn.Sequential(
                    nn.Conv1d(dim, dim*2, 3, dilation=2**d, padding=2**d),
                    nn.GLU(dim=1),
                ))
                self.blocks.append(block(dim))
                self.proj_out.append(nn.Conv1d(dim, 2*dim, 1))

    def forward(self, x: Float[Tensor, "B D L"], *args, **kwargs) -> Float[Tensor, "B D L"]:
        o = th.zeros_like(x)
        for proj_in, block, proj_out in zip(self.proj_in, self.blocks, self.proj_out):
            h = block(proj_in(x), *args, **kwargs)
            res, skip = proj_out(h).chunk(2, dim=1)
            x = (x + res) * 2 ** -0.5
            o = o + skip
        return o * len(self.blocks) ** -0.5