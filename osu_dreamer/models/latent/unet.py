
from dataclasses import dataclass
from jaxtyping import Float

from torch import nn, Tensor

from osu_dreamer.modules.res import Res
from osu_dreamer.modules.swiglu import SwiGLU
from osu_dreamer.modules.rms_norm import RMSNorm

@dataclass
class LayerArgs:
    n_layers: int
    expand: int
    radius: int

def Layer(dim: int, args: LayerArgs):
    return nn.Sequential(
        *( Res(
            RMSNorm(dim), 
            SwiGLU(dim, expand=args.expand, radius=args.radius), 
            RMSNorm(dim, gain=1e-3),
        ) for _ in range(args.n_layers) ), 
        RMSNorm(dim),
    ) # peri-ln

class UNetEncoder(nn.Module):
    def __init__(self, dim: int, n_downs: int, stride: int, args: LayerArgs):
        super().__init__()
        self.down = nn.AvgPool1d(stride)
        self.layers = nn.ModuleList([ Layer(dim, args) for _ in range(n_downs) ])
        self.unmixers = nn.ModuleList([ unmixer(dim, stride) for _ in range(n_downs) ])

    def forward(self, x: Float[Tensor, "B X L"]) -> tuple[ list[ Float[Tensor, "B X _l"] ], Float[Tensor, "B X l"] ]:
        skips = []
        for layer, unmix in zip(self.layers, self.unmixers):
            x = layer(x)
            skip, x = unmix(x)
            skips.append(skip)
            x = self.down(x)
        return skips, x
    
class UNetDecoder(nn.Module):
    def __init__(self, dim: int, n_downs: int, stride: int, args: LayerArgs):
        super().__init__()
        self.up = nn.Upsample(scale_factor=stride)
        self.layers = nn.ModuleList([ Layer(dim, args) for _ in range(n_downs) ])
        self.mixers = nn.ModuleList([ mixer(dim, stride) for _ in range(n_downs) ])

    def forward(self, skips: list[ Float[Tensor, "#B X _L"] ], x: Float[Tensor, "B X l"]) -> Float[Tensor, "B X L"]:
        for mix, layer in zip(self.mixers, self.layers):
            x = self.up(x)
            skip = skips.pop().expand(x.size(0), -1, -1)
            x = mix(skip, x)
            x = layer(x)
        return x
    
class unmixer(nn.Module):
    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.filter = nn.Conv1d(dim, dim, 1+2*(stride//2), 1, stride//2, groups=dim)
        self.proj = nn.Conv1d(dim, 2*dim, 1)

    def forward(
        self,
        x: Float[Tensor, "B D L"],
    ) -> tuple[
        Float[Tensor, "B D L"], # skip
        Float[Tensor, "B D L"], # x
    ]:
        return self.proj(self.filter(x)).chunk(2, dim=1)

class mixer(nn.Module):
    def __init__(self, dim: int, stride: int):
        super().__init__()
        self.filter = nn.Conv1d(dim, dim, 1+2*(stride//2), 1, stride//2, groups=dim)
        self.proj = nn.Conv1d(2*dim, dim, 1)

    def forward(
        self,
        skip: Float[Tensor, "B D L"],
        x: Float[Tensor, "B D L"],
    ) -> Float[Tensor, "B D L"]:
        import torch as th
        return self.proj(th.cat([skip, self.filter(x)], dim=1))