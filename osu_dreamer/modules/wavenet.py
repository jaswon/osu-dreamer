
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
    """wavenet receptive field: 1+s*(2**d-1)"""
    def __init__(
        self,
        dim: int,
        args: WaveNetArgs,
        block: Callable[[int], nn.Module],
    ):
        super().__init__()

        class layer(nn.Module):
            def __init__(self, depth: int):
                super().__init__()
                self.proj_x = nn.Sequential(
                    nn.GroupNorm(1, dim),
                    nn.Conv1d(dim, dim*2, 3, dilation=2**depth, padding=2**depth),
                    nn.GLU(dim=1),
                )
                self.block = block(dim)
                self.proj_out = nn.Conv1d(dim, 2*dim, 1)

            def forward(
                self, 
                x: Float[Tensor, "B D L"], 
                *args, **kwargs,
            ) -> tuple[
                Float[Tensor, "B D L"],
                Float[Tensor, "B D L"],
            ]:
                h = self.block(self.proj_x(x), *args, **kwargs)
                return self.proj_out(h).chunk(2, dim=1)

        self.layers = nn.ModuleList()
        for _ in range(args.num_stacks):
            for d in range(args.stack_depth):
                self.layers.append(layer(d))
        self.post_norm = nn.GroupNorm(1, dim)

    def forward(
        self, 
        x: Float[Tensor, "B D L"], 
        *args, **kwargs,
    ) -> Float[Tensor, "B D L"]:
        o = th.zeros_like(x)
        for layer in self.layers:
            res, skip = layer(x,*args,**kwargs)
            x = x + res
            o = o + skip
        return self.post_norm(o)