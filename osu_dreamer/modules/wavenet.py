
from typing import Callable
from jaxtyping import Float

from dataclasses import dataclass

import torch as th
from torch import nn, Tensor

class ResSkipNet(nn.Module):
    def __init__(
        self,
        dim: int,
        layers: list[nn.Module],
    ):
        super().__init__()
        self.norms = nn.ModuleList([ nn.GroupNorm(1, dim) for _ in layers ])
        self.layers = nn.ModuleList(layers)
        self.post_norm = nn.GroupNorm(1, dim)

    def forward(
        self, 
        x: Float[Tensor, "B X L"], 
        *args, **kwargs,
    ) -> Float[Tensor, "B X L"]:
        o = th.zeros_like(x)
        for norm, layer in zip(self.norms, self.layers):
            res, skip = layer(norm(x),*args,**kwargs).chunk(2, dim=1)
            x = x + res
            o = o + skip
        return self.post_norm(o)

@dataclass
class WaveNetArgs:
    num_stacks: int
    stack_depth: int

class WaveNet(ResSkipNet):
    """wavenet receptive field: 1+s*(2**d-1)"""
    def __init__(
        self,
        dim: int,
        args: WaveNetArgs,
        block: Callable[[int], nn.Module],
    ):
        class layer(nn.Module):
            def __init__(self, depth: int):
                super().__init__()
                self.proj_x = nn.Conv1d(dim, dim, 3, dilation=2**depth, padding=2**depth)
                self.block = block(depth)
                self.proj_out = nn.Conv1d(dim, 2*dim, 1)

            def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D*2 L"]:
                h = self.proj_x(x)
                h = self.block(h)
                return self.proj_out(h)

        super().__init__(dim, [
            layer(d)
            for _ in range(args.num_stacks)
            for d in range(args.stack_depth)
        ])