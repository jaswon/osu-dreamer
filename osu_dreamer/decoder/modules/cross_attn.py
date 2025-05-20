
from jaxtyping import Float, Int

from dataclasses import dataclass

from torch import nn, Tensor
import torch.nn.functional as F

from einops.layers.torch import Rearrange

from .rope import RoPE


@dataclass
class AttnArgs:
    head_dim: int
    n_heads: int
    kv_heads: int = -1

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

        kv_heads = args.kv_heads if args.kv_heads > 0 else args.n_heads
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
        x_t: Int[Tensor, "B Lq"],
        ctx: Float[Tensor, "B Lkv Dkv"], 
        ctx_t: Int[Tensor, "B Lkv"], 
    ) -> Float[Tensor, "B Lq Dq"]:
        q = self.q(x).float() # b h m d
        k,v = self.kv(ctx).float() # b h n d
        
        q = self.rope(q, x_t) * self.scale
        k = self.rope(k, ctx_t)

        # attn = linear_attn(q,k,v)
        attn = F.scaled_dot_product_attention(q,k,v, enable_gqa=True)
        return self.out(attn)