
from typing import Callable, Optional
from jaxtyping import Float

from dataclasses import dataclass

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

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
        y_dim: Optional[int],
        args: WaveNetArgs,
        block: Callable[[int], nn.Module],
    ):
        class layer(nn.Module):
            def __init__(self, depth: int):
                super().__init__()
                self.proj_y = None
                if y_dim is not None:
                    self.proj_y = nn.Sequential(
                        nn.Conv1d(y_dim, y_dim, 3,1,1, groups=y_dim),
                        nn.Conv1d(y_dim, dim*2, 1),
                    )

                self.proj_x = nn.Conv1d(dim, dim*2, 3, dilation=2**depth, padding=2**depth)
                self.block = block(depth)
                self.proj_out = nn.Conv1d(dim, 2*dim, 1)

            def forward(
                self, 
                x: Float[Tensor, "B D L"], 
                y: Optional[Float[Tensor, "B Y L"]],
                *args, **kwargs,
            ) -> Float[Tensor, "B D*2 L"]:
                h = self.proj_x(x)
                if self.proj_y is not None:
                    h = h + self.proj_y(y)
                h = F.glu(h, dim=1)
                h = self.block(h, *args, **kwargs)
                return self.proj_out(h)

        super().__init__(dim, [
            layer(d)
            for _ in range(args.num_stacks)
            for d in range(args.stack_depth)
        ])