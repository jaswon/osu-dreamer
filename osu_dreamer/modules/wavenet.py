
from typing import Callable
from jaxtyping import Float

from dataclasses import dataclass

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

@dataclass
class WaveNetArgs:
    num_stacks: int
    stack_depth: int

class WaveNet(nn.Module):
    """wavenet receptive field: 1+s*(2**d-1)"""
    def __init__(
        self,
        dim: int,
        y_dim: int,
        args: WaveNetArgs,
        block: Callable[[int], nn.Module],
    ):
        super().__init__()
        self.res_y = nn.ModuleList()
        self.proj_y = nn.ModuleList()
        self.proj_x = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.proj_out = nn.ModuleList()
        for _ in range(args.num_stacks):
            for d in range(args.stack_depth):
                self.res_y.append(nn.Sequential(
                    nn.GroupNorm(1, y_dim),
                    nn.Conv1d(y_dim, y_dim*2, 3, dilation=2**d, padding=2**d),
                    nn.GLU(dim=1),
                    nn.Conv1d(y_dim, y_dim, 1),
                ))
                self.proj_y.append(nn.Conv1d(y_dim, dim*2, 1))
                self.proj_x.append(nn.Sequential(
                    nn.GroupNorm(1, dim),
                    nn.Conv1d(dim, dim*2, 3, dilation=2**d, padding=2**d),
                ))
                self.blocks.append(block(dim))
                self.proj_out.append(nn.Conv1d(dim, 2*dim, 1))
        self.post_norm = nn.GroupNorm(1, dim)

    def forward(
        self, 
        x: Float[Tensor, "B D L"], 
        y: Float[Tensor, "B C L"],
        *args, **kwargs,
    ) -> Float[Tensor, "B D L"]:
        o = th.zeros_like(x)
        for res_y, proj_y, proj_x, block, proj_out in zip(self.res_y, self.proj_y, self.proj_x, self.blocks, self.proj_out):
            y = y + res_y(y)
            h = F.glu(proj_y(y) + proj_x(x), dim=1)
            h = block(h, *args, **kwargs)
            res, skip = proj_out(h).chunk(2, dim=1)
            x = x + res
            o = o + skip
        return self.post_norm(o)