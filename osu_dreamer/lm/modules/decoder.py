
from jaxtyping import Float, Int

from dataclasses import dataclass

from torch.utils.checkpoint import checkpoint
from torch import nn, Tensor

import xformers.ops as xops

from .attn import SelfAttention, CrossAttention, AttnKVCache

@dataclass
class DecoderArgs:
    n_heads: int
    n_layers: int
    dropout: float
    n_freqs: int = 64
    checkpoint: bool = True
    

DecoderLayerKVCache = tuple[
    AttnKVCache,    # self attention cache
    AttnKVCache,    # cross attention cache
]

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, ctx_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.self_attn = SelfAttention(d_model, n_heads, dropout)
        self.cross_attn = CrossAttention(d_model, n_heads, dropout, ctx_dim)
        self.ffn = xops.SwiGLU(d_model, (int(d_model * 8 / 3) + 7) // 8 * 8)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, 
        x: Float[Tensor, "B N D"], 
        emb_t: Int[Tensor, "B N"],
        ctx: Float[Tensor, "B L C"], 
        cache: DecoderLayerKVCache | None = None,
    ) -> tuple[Float[Tensor, "B N D"], DecoderLayerKVCache]:
        
        if cache is not None:
            sa_cache, ca_cache = cache
        else:
            sa_cache, ca_cache = None, None

        # Self attention
        sa_out, sa_cache = self.self_attn(self.norm1(x), cache=sa_cache)
        x = x + self.dropout(sa_out)
        
        # Cross attention
        ca_out, ca_cache = self.cross_attn(self.norm2(x), emb_t, ctx, cache=ca_cache)
        x = x + self.dropout(ca_out)
        
        # Feed forward
        ffn_out = self.ffn(self.norm3(x))
        x = x + self.dropout(ffn_out)
        
        return x, (sa_cache, ca_cache)

class Decoder(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        ctx_dim: int,
        args: DecoderArgs,
    ):
        super().__init__()
        
        if args.checkpoint:
            self.run_block = lambda block, *args: checkpoint(block, *args, use_reentrant=False)
        else:
            self.run_block = lambda block, *args: block(*args)

        self.layers = nn.ModuleList([
            DecoderLayer(emb_dim, ctx_dim, args.n_heads, args.dropout)
            for _ in range(args.n_layers)
        ])

    def forward(
        self,
        emb: Float[Tensor, "B N D"],
        emb_t: Int[Tensor, "B N"],
        ctx: Float[Tensor, "B L C"],
        cache: list[DecoderLayerKVCache] | None = None,
    ) -> tuple[Float[Tensor, "B N D"], list[DecoderLayerKVCache]]:
        x = emb
        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, new_layer_cache = self.run_block(layer, x, emb_t, ctx, layer_cache) # type: ignore
            new_caches.append(new_layer_cache)
            
        return x, new_caches