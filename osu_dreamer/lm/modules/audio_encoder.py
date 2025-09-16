
from jaxtyping import Float

from dataclasses import dataclass
 
import torch as th
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

from einops.layers.torch import Rearrange

import xformers.ops as xops

from osu_dreamer.lm.modules.attn import SelfAttention

@dataclass
class AudioEncoderArgs:
    depth: int
    n_heads: int
    dropout: float = 0.
    expand: int = 1
    checkpoint: bool = True
    freq_bins: int = 6

class AudioEncoder(nn.Module):
    def __init__(self, dim: int, args: AudioEncoderArgs, context_size: int):
        super().__init__()
        h_dim = dim * args.expand
        
        self.freq_proj = nn.Sequential(
            Rearrange('b f l -> b 1 f l'),
            nn.Conv2d(1, 1, (1, 7), (1,1), (1,3)),
            nn.SiLU(),
            nn.AdaptiveMaxPool2d((args.freq_bins, None)), # b 1 d l
            Rearrange('b 1 d l -> b l d'),
            nn.Linear(args.freq_bins, h_dim),
        )

        self.blocks = nn.ModuleList([
            nn.Sequential(
                AdaLNZero(h_dim, sequenceMixer(h_dim, args.n_heads, args.dropout, context_size)),
                AdaLNZero(h_dim, channelMixer(h_dim)),
            )
            for _ in range(args.depth)
        ])
        if args.checkpoint:
            self.run_block = lambda block, x: checkpoint(block, x, use_reentrant=False)
        else:
            self.run_block = lambda block, x: block(x)
        
        self.final = nn.Identity() if args.expand == 1 else nn.Linear(h_dim, dim)

    def forward(self, x: Float[Tensor, "B A L"]) -> Float[Tensor, "B L D"]:
        x = self.freq_proj(x)
        
        for block in self.blocks:
            x = self.run_block(block, x) # type: ignore

        return self.final(x)
    
class AdaLNZero(nn.Module):
    def __init__(self, dim: int, op: nn.Module):
        super().__init__()
        self.op = op
        self.norm = nn.RMSNorm(dim)
        self.scale = nn.Parameter(th.zeros(dim))
        self.shift = nn.Parameter(th.zeros(dim))
        self.alpha = nn.Parameter(th.zeros(dim))

    def forward(self, x: Float[Tensor, "B L D"]) -> Float[Tensor, "B L D"]:
        r = self.norm(x)
        r = r * (1+self.scale) + self.shift
        r = self.op(r)
        r = self.alpha * r
        return x + r

class sequenceMixer(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float, ctx_size: int):
        super().__init__()
        self.attn = SelfAttention(dim, n_heads, dropout)

    def forward(self, x: Float[Tensor, "B L D"]) -> Float[Tensor, "B L D"]:
        return self.attn(x)
        
    
class channelMixer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.ffn = xops.SwiGLU(dim, (int(dim * 8 / 3) + 7) // 8 * 8)

    def forward(self, x: Float[Tensor, "B L D"]) -> Float[Tensor, "B L D"]:
        return self.ffn(x)