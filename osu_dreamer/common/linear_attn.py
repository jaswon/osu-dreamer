
from jaxtyping import Float

from dataclasses import dataclass

import numpy as np
import torch as th
from torch import nn, Tensor

from einops.layers.torch import Rearrange
from einops import rearrange, repeat, pack, unpack


def rotate_half(x: Float[Tensor, "... D"]) -> Float[Tensor, "... D"]:
    x1, x2 = rearrange(x, '... (d r) -> ... d r', r = 2).unbind(dim = -1)
    x_r = th.stack((-x2, x1), dim = -1)
    return rearrange(x_r, '... d r -> ... (d r)')

class RoPE(nn.Module):
    def __init__(self, dim: int, max_timescale: float = 1e5, b=.7):
        super().__init__()
        d = dim // 2
        assert d * 2 == dim
        # a = np.log(max_timescale) / d**b
        # self.fs = th.exp(-a * (1 + th.arange(d, dtype=th.float)) ** b) * 2 * th.pi # D/2
        self.fs = max_timescale ** -(th.arange(0, dim, 2, dtype=th.float) / dim)

    def set_ts(self, ts: Float[Tensor, "B L"]):
        theta = (self.fs.to(ts.device) * ts[...,None]) # B L D/2
        theta = repeat(theta, '... d -> ... (d r)', r=2)
        self.sincos = th.stack([theta.sin(), theta.cos()]) # B L D

    def forward(self, x: Float[Tensor, "B ... D"]) -> Float[Tensor, "B ... D"]:
        sin, cos = self.sincos.to(x.device)
        return (
            th.einsum('b ... l d, b l d -> b ... l d', x, cos) +
            th.einsum('b ... l d, b l d -> b ... l d', rotate_half(x), sin)
        )
    

def exp_taylor_map(x: Float[Tensor, "... d"]) -> Float[Tensor, "... D"]:
    # exp(qkT) ~= 1 + qkT + (qkT)^2 / 2
    #           = 1 + qkT + qkTqkT / 2
    #           = 1 + qkT + qqTkkT / 2
    #           = [1,q,qqT/sqrt(2)] . [1,k,kkT/sqrt(2)]

    x, ps = pack([x], '* d')

    out, _ = pack([
        x.new_ones((x.shape[0],)), 
        x,
        th.einsum('b i, b j -> b i j', x, x) * 0.5 ** 0.5,
    ], 'b *')

    return unpack(out, ps, '* d')[0]

@dataclass
class AttnArgs:
    head_dim: int = 64
    n_heads: int = 16

# from taylor_series_linear_attention import TaylorSeriesLinearAttn
class LinearAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        rope: RoPE,
        args: AttnArgs,
    ):
        super().__init__()
        dim_inner = args.head_dim * args.n_heads
        self.scale = args.head_dim ** -0.5

        self.rope = rope

        self.qkv = nn.Sequential(
            nn.GroupNorm(1, dim),
            nn.Conv1d(dim, dim_inner * 3, 1, bias = False),
            Rearrange('b (qkv h d) n -> qkv b h n d', h = args.n_heads, qkv = 3),
        )

        self.out = nn.Sequential(
            Rearrange('b h n e -> b (h e) n'),
            nn.Conv1d(dim_inner, dim, 1, bias = False),
        )

    def forward(self, x: Float[Tensor, "B D L"], eps: float = 1e-5) -> Float[Tensor, "B D L"]:
        q, k, v = self.qkv(x) # b h n d

        L = x.size(-1)
        q = exp_taylor_map(self.rope(q) * self.scale * np.log(L)) / L # https://arxiv.org/abs/2202.12172
        k = exp_taylor_map(self.rope(k)) / L

        kv = th.einsum('b h n d, b h n e -> b h d e', k, v)
        qk_inv = th.einsum('b h n d, b h m d -> b h n', q, k).clamp(min = eps).pow(-1)
        attn = th.einsum('b h n d, b h d e, b h n -> b h n e', q, kv, qk_inv)

        return self.out(attn)