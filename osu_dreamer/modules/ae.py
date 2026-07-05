
from dataclasses import dataclass
from jaxtyping import Float

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
        self.downs = nn.ModuleList([
            nn.Identity() if i==0 else nn.Conv1d(dim, dim, 2+stride,stride,1) 
            for i in range(1+n_downs)
        ])
        self.layers = nn.ModuleList([
            Layer(dim, args.n_layers, args.radius) 
            for _ in range(1+n_downs)
        ])

    def forward(self, x: Float[Tensor, "B X L"]) -> list[ Float[Tensor, "B X _l"] ]:
        layers = []
        for down, layer in zip(self.downs, self.layers):
            x = down(x)
            x = layer(x)
            layers.append(x)

        return layers
    
class Encoder(UNetEncoder):
    def forward(self, x: Float[Tensor, "B X L"]) -> Float[Tensor, "B X _l"]:
        return super().forward(x)[-1]
    
class UNetDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        n_downs: int,
        stride: int,
        args: AEArgs,
    ):
        super().__init__()
        self.ups = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=stride),
                nn.Conv1d(dim, dim, 1+2*(stride//2+1),1,stride//2+1),
            )
            for _ in range(n_downs)
        ])
        self.layers = nn.ModuleList([ Layer(dim, args.n_layers, args.radius) for _ in range(1+n_downs) ])
        self.mixers = nn.ModuleList([ AdaLN1d(dim, dim) for _ in range(n_downs) ])

    def forward(self, xs: list[ Float[Tensor, "B X _L"] ]) -> Float[Tensor, "B X L"]:

        x = self.layers[0](xs.pop())
        for up, mix, layer in zip(self.ups, self.mixers, self.layers[1:]):
            x = layer(mix(up(x), xs.pop()))

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
