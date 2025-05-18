
from jaxtyping import Float

from dataclasses import dataclass

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from einops import pack, unpack

from .rope import RoPE


def exp_taylor_map(x: Float[Tensor, "... d"]) -> Float[Tensor, "... D"]:
    # exp(qkT) ~= 1 + qkT + (qkT)^2 / 2
    #           = 1 + qkT + qkTqkT / 2
    #           = 1 + qkT + qqTkkT / 2
    #           = [1,q,qqT/sqrt(2)] . [1,k,kkT/sqrt(2)]

    x, ps = pack([x], '* d')

    x0 = x.new_ones((x.shape[0],))
    x1 = x
    x2 = th.einsum('b i, b j -> b i j', x, x) * 0.5 ** 0.5

    # redundant values in even powers
    x2_diag = th.diagonal(x2, dim1=-2, dim2=-1)
    triu_mask = th.ones(x2.shape[-2:], dtype = th.bool).triu(1)
    x2_triu = x2[:,triu_mask] * 2 ** 0.5
    x2 = th.cat([x2_diag, x2_triu], dim=-1)

    out, _ = pack([ x0, x1, x2 ], 'b *')

    # D = 1 + d + d*(d+1)/2 = (d+1)*(d/2+1)
    return unpack(out, ps, '* d')[0]


def linear_attn(
    q: Float[Tensor, "B H Lq Dqk"],
    k: Float[Tensor, "B #H Lkv Dqk"],
    v: Float[Tensor, "B #H Lkv Dv"],
    eps: float = 1e-6,
) -> Float[Tensor, "B H Lq Dv"]:
    q = exp_taylor_map(q)
    k = exp_taylor_map(k) 
    
    kv = th.einsum('b h n d, b h n e -> b h d e', k, v)
    qk_inv = th.einsum('b h m d, b h n d -> b h m', q, k).clamp(min = eps).pow(-1)
    return th.einsum('b h m d, b h d e, b h m -> b h m e', q, kv, qk_inv)


@dataclass
class AttnArgs:
    head_dim: int
    n_heads: int
    one_kv_head: bool = True

# from taylor_series_linear_attention import TaylorSeriesLinearAttn
class CrossAttn(nn.Module):
    def __init__(
        self,
        dim: int,
        ctx_dim: int,
        rope: RoPE,
        args: AttnArgs,
    ):
        super().__init__()
        dim_inner = args.head_dim * args.n_heads
        self.scale = args.head_dim ** -0.5

        assert rope.dim == args.head_dim
        self.rope = rope

        self.q = nn.Sequential(
            nn.Linear(dim, dim_inner, bias=False),
            Rearrange('b m (h d) -> b h m d', h = args.n_heads),
        )

        kv_heads = 1 if args.one_kv_head else args.n_heads
        self.kv = nn.Sequential(
            nn.Linear(ctx_dim, 2 * kv_heads * args.head_dim, bias=False),
            Rearrange('b n (kv h d) -> kv b h n d', h = kv_heads, kv = 2),
        )

        self.out = nn.Sequential(
            Rearrange('b h m e -> b m (h e)'),
            nn.Linear(dim_inner, dim, bias=False),
        )

    def forward(
        self, 
        x: Float[Tensor, "B Lq Dq"], 
        x_t: Float[Tensor, "B Lq"],
        ctx: Float[Tensor, "B Lkv Dkv"], 
        ctx_t: Float[Tensor, "B Lkv"], 
    ) -> Float[Tensor, "B Lq Dq"]:
        q = self.q(x).float() # b h m d
        k,v = self.kv(ctx).float() # b h n d
        
        q = self.rope(q, x_t) * self.scale
        k = self.rope(k, ctx_t)

        # attn = linear_attn(q,k,v)
        attn = F.scaled_dot_product_attention(q,k,v, enable_gqa=True)
        return self.out(attn)