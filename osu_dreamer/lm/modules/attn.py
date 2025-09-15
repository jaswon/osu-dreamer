
from jaxtyping import Float, Int

import torch as th
from torch import nn, Tensor

import xformers.ops as xops


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.inv_freq = 1.0 / (10000 ** (th.arange(0, dim, 2).float() / dim))

    def forward(
        self, 
        x: Float[Tensor, "B N H D"], 
        t: Int[Tensor, "B N"] | None = None,
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
    
AttnKVCache = tuple[
    Float[Tensor, "B _ h d"],  # keys
    Float[Tensor, "B _ h d"],  # values
]

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.rotary_emb = RotaryEmbedding(self.d_head)
        self.dropout = dropout

    def forward(self, x: Float[Tensor, "B N D"]) -> Float[Tensor, "B N D"]:
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        
        q = q.unflatten(2, (self.n_heads, -1)) # B N H d
        k = k.unflatten(2, (self.n_heads, -1)) # B L H d
        v = v.unflatten(2, (self.n_heads, -1)) # B L H d

        q = self.rotary_emb(q)  
        k = self.rotary_emb(k)

        out = xops.memory_efficient_attention(q, k, v)
        out = out.flatten(-2, -1)  # B N D
        
        return self.out_proj(out)

class CausalSelfAttention(nn.Module):
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
        cache: AttnKVCache | None = None,
    ) -> tuple[Float[Tensor, "B N D"], AttnKVCache]:
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
        cache: AttnKVCache | None = None,
    ) -> tuple[Float[Tensor, "B N D"], AttnKVCache]:
        
        q = self.q_proj(x) # B N D
        q = q.unflatten(2, (self.n_heads, -1)) # B N H d
        q = self.rotary_emb(q, t=emb_t)

        if cache is not None:
            k, v = cache
        else:
            # Compute k,v from context - this only happens once per sequence
            k, v = self.kv_proj(ctx).chunk(2, dim=-1)  # B L D
            
            k = k.unflatten(2, (self.n_heads, -1)) # B L H d
            v = v.unflatten(2, (self.n_heads, -1)) # B L H d

            k = self.rotary_emb(k)
        
        attn_out = xops.memory_efficient_attention(q, k, v)  # B N H d
        attn_out = attn_out.flatten(-2, -1)  # B N D
        attn_out = self.out_proj(attn_out)
        
        # Apply gating
        gate_val = th.sigmoid(self.gate(x))
        return gate_val * attn_out, (k, v)