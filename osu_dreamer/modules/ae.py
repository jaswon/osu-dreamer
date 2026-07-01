
from dataclasses import dataclass
from jaxtyping import Float

from torch import nn, Tensor
import torch.nn.functional as F

from einops import repeat

from osu_dreamer.modules.res import Res
from osu_dreamer.modules.swiglu import SwiGLU
from osu_dreamer.modules.rms_norm import RMSNorm

Layer = lambda d_h, n: nn.Sequential(*(
    layer for _ in range(n)
    for layer in [ Res(RMSNorm(d_h), SwiGLU(d_h), alpha=n), RMSNorm(d_h) ] # keel
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
            nn.Conv1d(args.h_dim, d_emb, 1) if d_emb > 0 else nn.Identity(),
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
        self.stride = stride
        self.chunk_size = stride ** n_downs

        self.proj_in = nn.Conv1d(d_a, args.h_dim, 1) if d_a != args.h_dim else nn.Identity()
        self.proj_out = nn.Conv1d(args.h_dim, d_x, 1) if d_x != args.h_dim else nn.Identity()

        self.downs = nn.ModuleList([ Layer(args.h_dim, args.n_layers) for _ in range(n_downs) ])
        self.ups = nn.ModuleList([ Layer(args.h_dim, args.n_layers) for _ in range(n_downs) ])

        self.mix = AdaLN1d(args.h_dim, d_emb)

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

        a = self.proj_in(a)

        fs = []
        for down in self.downs:
            a_h = down(a)
            a = a_h.unflatten(-1, (-1, self.stride)).mean(dim=-1)
            fs.append(a_h - repeat(a, 'b d l -> b d (l r)', r=self.stride))
        
        h = self.mix(a, h)

        for up in self.ups:
            h = up(fs.pop() + repeat(h, 'b d l -> b d (l r)', r=self.stride))

        return self.proj_out(h)[:,:,:L]

    
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
