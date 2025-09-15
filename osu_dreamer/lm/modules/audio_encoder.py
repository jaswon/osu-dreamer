
from jaxtyping import Float

from dataclasses import dataclass
 
import torch as th
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

from einops.layers.torch import Rearrange

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.mingru import MinGRU

@dataclass
class AudioEncoderArgs:
    depth: int
    expand: int = 1
    checkpoint: bool = True
    freq_bins: int = 6

class AudioEncoder(nn.Module):
    def __init__(self, dim: int, args: AudioEncoderArgs):
        super().__init__()
        h_dim = dim * args.expand
        
        self.freq_proj = nn.Sequential(
            Rearrange('b f l -> b 1 f l'),
            nn.Conv2d(1, 1, (1, 7), (1,1), (1,3)),
            nn.SiLU(),
            nn.AdaptiveMaxPool2d((args.freq_bins, None)), # b 1 d l
            Rearrange('b 1 d l -> b d l'),
            nn.Conv1d(args.freq_bins, h_dim, 1),
        )

        self.blocks = nn.ModuleList([
            nn.Sequential(
                AdaLNZero(h_dim, sequenceMixer(h_dim)),
                AdaLNZero(h_dim, channelMixer(h_dim)),
            )
            for _ in range(args.depth)
        ])
        if args.checkpoint:
            self.run_block = lambda block, x: checkpoint(block, x, use_reentrant=False)
        else:
            self.run_block = lambda block, x: block(x)
        
        self.final = nn.Sequential(
            Rearrange('b h l -> b l h'),
            nn.Identity() if args.expand == 1 else nn.Linear(h_dim, dim),
        )

    def forward(self, x: Float[Tensor, "B A L"]) -> Float[Tensor, "B L D"]:
        x = self.freq_proj(x)
        
        for block in self.blocks:
            x = self.run_block(block, x) # type: ignore

        return self.final(x)
    
class AdaLNZero(nn.Module):
    def __init__(self, dim: int, op: nn.Module):
        super().__init__()
        self.op = op
        self.scale = nn.Parameter(th.zeros(dim, 1))
        self.shift = nn.Parameter(th.zeros(dim, 1))
        self.alpha = nn.Parameter(th.zeros(dim, 1))

    def forward(self, x: Float[Tensor, "B X L"]) -> Float[Tensor, "B X L"]:
        r = MP.pixel_norm(x)
        r = r * (1+self.scale) + self.shift
        r = self.op(r)
        r = self.alpha * r
        return x + r

class sequenceMixer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        h_dim = dim // 2
        assert h_dim * 2 == dim
        self.fore = MinGRU(dim, h_dim)
        self.back = MinGRU(dim, h_dim)

    def forward(self, x: Float[Tensor, "B X L"]) -> Float[Tensor, "B X L"]:
        return MP.cat([
            self.fore(x), 
            self.back(x.flip(2)).flip(2),
        ], dim=1)
    
class channelMixer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj_h = MP.Conv1d(dim, dim, 1)
        self.proj_g = MP.Conv1d(dim, dim, 1)
        self.proj_out = MP.Conv1d(dim, dim, 1)

    def forward(self, x: Float[Tensor, "B X L"]) -> Float[Tensor, "B X L"]:
        h,g = self.proj_h(x), self.proj_g(x)
        return self.proj_out(h*MP.silu(g))