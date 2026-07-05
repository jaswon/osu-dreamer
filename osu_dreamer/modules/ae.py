
from dataclasses import dataclass
from jaxtyping import Float

from torch import nn, Tensor
import torch.nn.functional as F

from einops import repeat

from osu_dreamer.modules.res import Res
from osu_dreamer.modules.swiglu import SwiGLU
from osu_dreamer.modules.rms_norm import RMSNorm

Layer = lambda d_h, n, r: nn.Sequential(*(
    Res(RMSNorm(d_h), SwiGLU(d_h, radius=r), RMSNorm(d_h)) # peri-ln
    for _ in range(n)
), RMSNorm(d_h))

@dataclass
class AEArgs:
    n_layers: int
    radius: int = 1

class Encoder(nn.Module):
    def __init__(
        self,
        dim: int,
        n_downs: int,
        stride: int,
        args: AEArgs,
    ):
        super().__init__()
        self.chunk_size = stride ** n_downs

        self.downs = nn.ModuleList([ nn.Conv1d(dim, dim, 2+stride,stride,1) for _ in range(n_downs)])
        self.layers = nn.ModuleList([ Layer(dim, args.n_layers, args.radius) for _ in range(n_downs) ])

    def forward(
        self, 
        x: Float[Tensor, "B X L"],
    ) -> list[ Float[Tensor, "B X _l"] ]:
        c = self.chunk_size
        pad = (c-x.size(-1)%c)%c
        if pad > 0:
            x = F.pad(x, (0, pad), mode='replicate')

        layers = [x]
        for down, layer in zip(self.downs, self.layers):
            x = down(x)
            x = layer(x)
            layers.append(x)

        return layers
    
class Decoder(nn.Module):
    def __init__(
        self,
        dim: int,
        n_downs: int,
        stride: int,
        args: AEArgs,
    ):
        super().__init__()
        self.stride = stride

        self.layers = nn.ModuleList([ Layer(dim, args.n_layers, args.radius) for _ in range(n_downs) ])
        self.mixers = nn.ModuleList([ AdaLN1d(dim, dim) for _ in range(n_downs) ])

    def forward(
        self,
        xs: list[ Float[Tensor, "B X _L"] ],
    ) -> Float[Tensor, "B X L"]:

        x = xs.pop()
        for mix, layer in zip(self.mixers, self.layers):
            x = repeat(x, 'b d l -> b d (l r)', r=self.stride)
            x = mix(xs.pop(), x)
            x = layer(x)

        return x

    
class AdaLN1d(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = RMSNorm(dim, affine=False)
        self.proj = nn.Conv1d(cond_dim, dim*2, 1)

    def forward(
        self,
        x: Float[Tensor, "B X L"],
        c: Float[Tensor, "B C L"],
    ) -> Float[Tensor, "B X L"]:
        scale, shift = self.proj(c).chunk(2, dim=1)
        return self.norm(x) * (1 + scale) + shift
