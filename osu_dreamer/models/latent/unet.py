
from dataclasses import dataclass
from jaxtyping import Float

from torch import nn, Tensor

from osu_dreamer.common.swiglu import SwiGLU
from osu_dreamer.common.rms_norm import RMSNorm

@dataclass
class LayerArgs:
    n_layers: int
    expand: int
    radius: int

def zero(m: nn.Linear):
    nn.init.zeros_(m.weight)
    nn.init.zeros_(m.bias)
    return m

class layer(nn.Module):
    def __init__(self, dim: int, cond_dim: int, args: LayerArgs):
        super().__init__()
        self.norms = nn.ModuleList([ RMSNorm(dim) for _ in range(args.n_layers) ])
        self.blocks = nn.ModuleList([
            nn.Sequential(
                SwiGLU(dim, expand=args.expand, radius=args.radius), 
                RMSNorm(dim, gain=1e-3),
            )
            for _ in range(args.n_layers)
        ])
        self.out_norm = RMSNorm(dim)
    
        self.films = None
        if cond_dim > 0:
            self.films = nn.ModuleList([ zero(nn.Linear(cond_dim, 2*dim)) for _ in range(args.n_layers) ])

    def forward(
        self,
        x: Float[Tensor, "B X L"],
        cond: None | Float[Tensor, "B C"] = None,
    ) -> Float[Tensor, "B X L"]:
        if self.films is not None:
            assert cond is not None, "conditional layer requires `cond`"
            films = [ film(cond)[:,:,None].chunk(2, dim=1) for film in self.films ]
        else:
            assert cond is None, "conditioning passed to an unconditional layer"
            films = [ (0,0) ] * len(self.blocks)

        for (scale, shift), norm, block in zip(films, self.norms, self.blocks):
            x = x + block(norm(x) * (1 + scale) + shift)
        return self.out_norm(x)

class UNetEncoder(nn.Module):
    def __init__(self, dim: int, n_downs: int, stride: int, args: LayerArgs):
        super().__init__()
        self.downs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, 1+2*(stride//2), 1, stride//2, groups=dim),
                nn.AvgPool1d(stride),
            )
            for _ in range(n_downs)
        ])
        self.layers = nn.ModuleList([ layer(dim, 0, args) for _ in range(n_downs) ])
        self.unmixers = nn.ModuleList([ unmixer(dim) for _ in range(n_downs) ])

    def forward(self, x: Float[Tensor, "B X L"]) -> tuple[ list[ Float[Tensor, "B X _l"] ], Float[Tensor, "B X l"] ]:
        skips = []
        for layer, unmix, down in zip(self.layers, self.unmixers, self.downs):
            x = layer(x)
            skip, x = unmix(x)
            skips.append(skip)
            x = down(x)
        return skips, x
    
class UNetDecoder(nn.Module):
    def __init__(self, dim: int, cond_dim: int, n_downs: int, stride: int, args: LayerArgs):
        super().__init__()
        self.ups = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=stride),
                nn.Conv1d(dim, dim, 1+2*(stride//2), 1, stride//2, groups=dim),
            )
            for _ in range(n_downs)
        ])
        self.layers = nn.ModuleList([ layer(dim, cond_dim, args) for _ in range(n_downs) ])
        self.mixers = nn.ModuleList([ mixer(dim) for _ in range(n_downs) ])

    def forward(
        self,
        skips: list[ Float[Tensor, "#B X _L"] ],
        x: Float[Tensor, "B X l"],
        cond: None | Float[Tensor, "B C"] = None,
    ) -> Float[Tensor, "B X L"]:
        for up, mix, layer in zip(self.ups, self.mixers, self.layers):
            x = up(x)
            skip = skips.pop().expand(x.size(0), -1, -1)
            x = mix(skip, x)
            x = layer(x, cond)
        return x
    
class unmixer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        
    def forward(
        self,
        x: Float[Tensor, "B D L"],
    ) -> tuple[
        Float[Tensor, "B D L"], # skip
        Float[Tensor, "B D L"], # x
    ]:
        return x, x

class mixer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        
    def forward(
        self,
        skip: Float[Tensor, "B D L"],
        x: Float[Tensor, "B D L"],
    ) -> Float[Tensor, "B D L"]:
        return skip + x