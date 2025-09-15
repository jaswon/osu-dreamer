
from typing import Optional
from jaxtyping import Float, Int

from dataclasses import dataclass

import torch as th
from torch.utils.checkpoint import checkpoint
from torch import nn, Tensor

import xformers.ops as xops


@dataclass
class DecoderArgs:
    n_heads: int
    n_layers: int
    dropout: float
    n_freqs: int = 64
    checkpoint: bool = True

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.inv_freq = 1.0 / (10000 ** (th.arange(0, dim, 2).float() / dim))

    def forward(
        self, 
        x: Float[Tensor, "B N H D"], 
        t: Optional[Int[Tensor, "B N"]] = None,
        offset: int = 0,
    ) -> Float[Tensor, "B N H D"]:
        if t is None:
            t = th.arange(x.size(1), device=x.device, dtype=self.inv_freq.dtype)[None] + offset
        freqs = th.einsum("bn,f->bnf", t, self.inv_freq.to(x.device))
        emb = th.cat((freqs, freqs), dim=2)[:,:,None,:] # B N 1 D

        def rotate_half(x: Tensor) -> Tensor:
            x1, x2 = x.chunk(2, dim=-1)
            return th.cat((-x2, x1), dim=-1)

        return (x * emb.cos()) + (rotate_half(x) * emb.sin())

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, max_cache_len: int = 1024):
        super().__init__()
        self.max_cache_len = max_cache_len
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary_emb = RotaryEmbedding(self.d_head)
        self.dropout = dropout

    def forward(
        self, 
        x: Float[Tensor, "B N D"], 
        cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Float[Tensor, "B N D"], tuple[Tensor, Tensor]]:
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        
        q = q.unflatten(2, (self.n_heads, -1)) # B N H d
        k = k.unflatten(2, (self.n_heads, -1)) # B L H d
        v = v.unflatten(2, (self.n_heads, -1)) # B L H d
        
        offset = 0
        if cache is not None:
            past_k, past_v = cache
            offset = past_k.shape[1]
            k = th.cat([past_k, k], dim=1)
            v = th.cat([past_v, v], dim=1)

            if k.shape[1] > self.max_cache_len:
                k = k[:, -self.max_cache_len:]
                v = v[:, -self.max_cache_len:]

        q = self.rotary_emb(q, offset=offset)  
        k = self.rotary_emb(k, offset=offset)

        attn_bias = xops.LowerTriangularMask() if cache is None else None
        out = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        out = out.flatten(-2, -1)  # B N D
        
        return self.out_proj(out), (k, v)

class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, ctx_dim: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.kv_proj = nn.Linear(ctx_dim, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, d_model)
        self.rotary_emb = RotaryEmbedding(self.d_head)
        
        self.dropout = dropout
        
    def forward(
        self, 
        x: Float[Tensor, "B N D"], 
        emb_t: Int[Tensor, "B N"],
        ctx: Float[Tensor, "B L C"],
    ) -> Float[Tensor, "B N D"]:
        q = self.q_proj(x) # B N D
        k, v = self.kv_proj(ctx).chunk(2, dim=-1)  # B L D
        
        q = q.unflatten(2, (self.n_heads, -1)) # B N H d
        k = k.unflatten(2, (self.n_heads, -1)) # B L H d
        v = v.unflatten(2, (self.n_heads, -1)) # B L H d

        q = self.rotary_emb(q, t=emb_t)
        k = self.rotary_emb(k)
        
        attn_out = xops.memory_efficient_attention(q, k, v)  # B N H d
        attn_out = attn_out.flatten(-2, -1)  # B N D
        attn_out = self.out_proj(attn_out)
        
        # Apply gating
        gate_val = th.sigmoid(self.gate(x))
        return gate_val * attn_out

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
        cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Float[Tensor, "B N D"], tuple[Tensor, Tensor]]:
        # Self attention
        sa_out, new_cache = self.self_attn(self.norm1(x), cache=cache)
        x = x + self.dropout(sa_out)
        
        # Cross attention
        ca_out = self.cross_attn(self.norm2(x), emb_t, ctx)
        x = x + self.dropout(ca_out)
        
        # Feed forward
        ffn_out = self.ffn(self.norm3(x))
        x = x + self.dropout(ffn_out)
        
        return x, new_cache

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
        cache: list[tuple[Tensor, Tensor]] | None = None,
    ) -> tuple[Float[Tensor, "B N D"], list[tuple[Tensor, Tensor]]]:
        x = emb
        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, new_layer_cache = self.run_block(layer, x, emb_t, ctx, layer_cache) # type: ignore
            new_caches.append(new_layer_cache)
            
        return x, new_caches