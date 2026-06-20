
from dataclasses import dataclass
from jaxtyping import Float

from torch import nn, Tensor
import torch.nn.functional as F

from osu_dreamer.modules.res import Res
from osu_dreamer.modules.derf import Derf
from osu_dreamer.modules.swiglu import SwiGLU

from torch import nn
import torch.nn.functional as F

Layer = lambda d_h, n: nn.Sequential(*(
    Res(Derf(d_h, 1), SwiGLU(d_h))
    for _ in range(n)
))

@dataclass
class AEArgs:
    h_dim: int
    n_layers: int

class Encoder(nn.Module):
    def __init__(
        self,
        d_x: int,
        d_emb: int,
        n_downs: int,
        stride: int,
        args: AEArgs,
    ):
        super().__init__()
        self.chunk_size = stride ** n_downs
        self.net = nn.Sequential(
            nn.Conv1d(d_x, args.h_dim, 1) if d_x != args.h_dim else nn.Identity(),
            *(
                layer for _ in range(n_downs)
                for layer in [
                    Layer(args.h_dim, args.n_layers),
                    nn.Conv1d(args.h_dim, args.h_dim, 2+stride,stride,1),
                ]
            ),
            nn.Conv1d(args.h_dim, d_emb, 1) if args.h_dim != d_emb else nn.Identity(),
            nn.GroupNorm(1, d_emb, affine=False),
        )

    def forward(self, x: Float[Tensor, "B X L"]) -> Float[Tensor, "B E l"]:
        c = self.chunk_size
        pad = (c-x.size(-1)%c)%c
        if pad > 0:
            x = F.pad(x, (0, pad), mode='replicate')
        return self.net(x)
    
class Decoder(nn.Module):
    def __init__(
        self,
        d_a: int,
        d_x: int,
        d_emb: int,
        n_downs: int,
        stride: int,
        args: AEArgs,
    ):
        super().__init__()
        self.chunk_size = stride ** n_downs

        self.downs = nn.ModuleList([
            nn.Sequential(
                Layer(d_a, args.n_layers),
                nn.Conv1d(d_a, d_a, 2+stride,stride,1),
            )
            for _ in range(n_downs-1)
        ])

        self.proj_in = nn.Conv1d(d_emb, args.h_dim, 1) if d_emb != args.h_dim else nn.Identity()
        self.proj_out = nn.Conv1d(args.h_dim, d_x, 1)

        self.ups = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=stride),
                nn.Conv1d(args.h_dim, args.h_dim, 1+2*(stride//2+1),1,stride//2+1),
            )
            for _ in range(n_downs)
        ])

        self.mixes = nn.ModuleList([
            AudioFiLM(args.h_dim, d_a)
            for _ in range(n_downs)
        ])

        self.layers = nn.ModuleList([
            Layer(args.h_dim, args.n_layers)
            for _ in range(n_downs)
        ])

    def forward(
        self,
        a: Float[Tensor, "B F L"],
        h: Float[Tensor, "B E l"],
    ) -> Float[Tensor, "B X L"]:
        c = self.chunk_size
        L = a.size(-1)
        pad = (c-L%c)%c
        if pad > 0:
            a = F.pad(a, (0, pad), mode='replicate')

        fs = [a]
        for down in self.downs:
            fs.append(down(fs[-1]))
        
        h = self.proj_in(h)
        for up, mix, layer in zip(self.ups, self.mixes, self.layers):
            h = layer(mix(up(h), fs.pop()))
        return self.proj_out(h)[:,:,:L]

    
class AudioFiLM(nn.Module):
    def __init__(self, dim: int, audio_dim: int):
        super().__init__()
        self.norm = Derf(dim, 1)
        self.proj = nn.Conv1d(audio_dim, dim*2, 1)

    def forward(
        self,
        x: Float[Tensor, "B D L"],
        a: Float[Tensor, "B A L"],
    ) -> Float[Tensor, "B D L"]:
        scale, shift = self.proj(a).chunk(2, dim=1)
        return self.norm(x) * (1 + scale) + shift
