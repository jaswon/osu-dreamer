from __future__ import annotations
from jaxtyping import Float

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
    checkpoint: bool = True

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (th.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq); self.inv_freq: Tensor

    def forward(self, x: Float[Tensor, "... N D"], offset: int = 0) -> Float[Tensor, "... N D"]:
        seq_len = x.shape[-2]
        t = th.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype) + offset
        freqs = th.einsum("i,j->ij", t, self.inv_freq)
        emb = th.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos()
        sin = emb.sin()

        def rotate_half(x: Tensor) -> Tensor:
            x1, x2 = x.chunk(2, dim=-1)
            return th.cat((-x2, x1), dim=-1)

        return (x * cos) + (rotate_half(x) * sin)

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
        B, N, _ = x.shape
        
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        
        q = q.view(B, N, self.n_heads, self.d_head)
        k = k.view(B, N, self.n_heads, self.d_head)
        v = v.view(B, N, self.n_heads, self.d_head)
        
        offset = 0
        if cache is not None:
            past_k, past_v = cache
            offset = past_k.shape[1]
            k = th.cat([past_k, k], dim=1)
            v = th.cat([past_v, v], dim=1)

            if k.shape[1] > self.max_cache_len:
                k = k[:, -self.max_cache_len:]
                v = v[:, -self.max_cache_len:]

        q = self.rotary_emb(q.transpose(1, 2), offset=offset).transpose(1, 2)  
        k = self.rotary_emb(k.transpose(1, 2), offset=offset).transpose(1, 2)

        attn_bias = xops.LowerTriangularMask() if cache is None else None
        out = xops.memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        
        out = out.view(B, N, -1)
        
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
        
        self.dropout = dropout
        
    def forward(
        self, 
        x: Float[Tensor, "B N D"], 
        ctx: Float[Tensor, "B L C"]
    ) -> Float[Tensor, "B N D"]:
        B, N, D = x.shape
        B_ctx, L, C = ctx.shape
        
        # Project queries from x
        q = self.q_proj(x)  # B N D
        q = q.view(B, N, self.n_heads, self.d_head)  # B N H d
        
        # Project keys and values from context
        k, v = self.kv_proj(ctx).chunk(2, dim=-1)  # B L D each
        k = k.view(B, L, self.n_heads, self.d_head)  # B L H d
        v = v.view(B, L, self.n_heads, self.d_head)  # B L H d
        
        attn_out = xops.memory_efficient_attention(q, k, v)  # B N H d
        
        # Reshape and project output
        attn_out = attn_out.view(B, N, D)  # B N D
        attn_out = self.out_proj(attn_out)
        
        # Apply gating
        gate_val = th.sigmoid(self.gate(x))
        return gate_val * attn_out

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float, ctx_dim: int):
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
        ctx: Float[Tensor, "B L C"], 
        cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Float[Tensor, "B N D"], tuple[Tensor, Tensor]]:
        # Self attention
        sa_out, new_cache = self.self_attn(self.norm1(x), cache=cache)
        x = x + self.dropout(sa_out)
        
        # Cross attention
        ca_out = self.cross_attn(self.norm2(x), ctx)
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
            DecoderLayer(emb_dim, args.n_heads, args.dropout, ctx_dim)
            for _ in range(args.n_layers)
        ])

    def forward(
        self,
        embs: Float[Tensor, "B N D"],
        ctx: Float[Tensor, "B L C"],
        cache: list[tuple[Tensor, Tensor]] | None = None,
    ) -> tuple[Float[Tensor, "B N D"], list[tuple[Tensor, Tensor]]]:
        x = embs
        new_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = cache[i] if cache is not None else None
            x, new_layer_cache = self.run_block(layer, x, ctx, layer_cache) # type: ignore
            new_caches.append(new_layer_cache)
            
        return x, new_caches