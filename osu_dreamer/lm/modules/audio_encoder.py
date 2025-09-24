
from jaxtyping import Float

from dataclasses import dataclass

import math
  
import torch as th
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

from einops.layers.torch import Rearrange

import xformers.ops as xops

from osu_dreamer.data.load_audio import A_DIM
from osu_dreamer.lm.modules.attn import SelfAttention


def create_sinusoidal_embeddings(length: int, dim: int) -> Float[Tensor, "{length} {dim}"]:
    pos_emb = th.zeros(length, dim)
    positions = th.arange(0, length, dtype=th.float).unsqueeze(1)
    div_term = th.exp(th.arange(0, dim, 2, dtype=th.float) * -(math.log(10000.0) / dim))
    pos_emb[:, 0::2] = th.sin(positions * div_term)
    pos_emb[:, 1::2] = th.cos(positions * div_term)
    return pos_emb

@dataclass
class AudioEncoderArgs:
    depth: int
    n_heads: int
    dropout: float = 0.
    expand: int = 1
    checkpoint: bool = True

class AudioEncoder(nn.Module):
    def __init__(self, dim: int, args: AudioEncoderArgs, context_size: int):
        super().__init__()
        h_dim = dim * args.expand

        self.pos_emb: th.Tensor
        self.register_buffer('pos_emb', create_sinusoidal_embeddings(context_size, h_dim))
        
        self.stem = nn.Sequential(
            nn.Conv1d(A_DIM, A_DIM, 3,1,1, groups=A_DIM),
            nn.Conv1d(A_DIM, h_dim, 1),
            nn.GELU(),
            nn.Conv1d(h_dim, h_dim, 3,1,1, groups=h_dim),
            nn.Conv1d(h_dim, h_dim, 1),
            nn.GELU(),
            Rearrange('b d l -> b l d'),
        )

        self.blocks = nn.ModuleList([
            nn.Sequential(
                AdaLNZero(h_dim, sequenceMixer(h_dim, args.n_heads, args.dropout)),
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
        x = self.stem(x)
        x = x + self.pos_emb[:x.size(1)]
        
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
    def __init__(self, dim: int, n_heads: int, dropout: float):
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