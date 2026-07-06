
from dataclasses import dataclass
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.modules.res import Res
from osu_dreamer.modules.swiglu import SwiGLU
from osu_dreamer.modules.rms_norm import RMSNorm

Layer = lambda d_h, n, r: nn.Sequential(RMSNorm(d_h), *(
    Res(RMSNorm(d_h), SwiGLU(d_h, radius=r), RMSNorm(d_h, gain=1e-3))
    for _ in range(n)
), RMSNorm(d_h)) # peri-ln

@dataclass
class AEArgs:
    n_layers: int
    radius: int = 1

class UNetEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        n_downs: int,
        stride: int,
        args: AEArgs,
    ):
        super().__init__()
        self.down = nn.AvgPool1d(stride)
        self.layers = nn.ModuleList([ Layer(dim, args.n_layers, args.radius) for _ in range(n_downs) ])
        self.unmixers = nn.ModuleList([ nn.Conv1d(dim, 2*dim, 1) for _ in range(n_downs) ])

    def forward(self, x: Float[Tensor, "B X L"]) -> tuple[ list[ Float[Tensor, "B X _l"] ], Float[Tensor, "B X l"] ]:
        skips = []
        for layer, unmixer in zip(self.layers, self.unmixers):
            x = layer(x)
            skip, x = unmixer(x).chunk(2, dim=1)
            skips.append(skip)
            x = self.down(x)
        return skips, x
    
class UNetDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        n_downs: int,
        stride: int,
        args: AEArgs,
    ):
        super().__init__()
        self.up = nn.Upsample(scale_factor=stride)
        self.layers = nn.ModuleList([ Layer(dim, args.n_layers, args.radius) for _ in range(n_downs) ])
        self.mixers = nn.ModuleList([ nn.Conv1d(2*dim, dim, 1) for _ in range(n_downs) ])

    def forward(self, skips: list[ Float[Tensor, "#B X _L"] ], x: Float[Tensor, "B X l"]) -> Float[Tensor, "B X L"]:
        for mix, layer in zip(self.mixers, self.layers):
            x = self.up(x)
            skip = skips.pop().expand(x.size(0), -1, -1)
            x = mix(th.cat([x, skip], dim=1))
            x = layer(x)
        return x