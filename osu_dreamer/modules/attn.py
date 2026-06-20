
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from einops import rearrange


class RoPE(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.register_buffer("freqs", th.empty(()), persistent=False)
        self.freqs: Float[th.Tensor, "N D"]
        self._init_freq(2048)

    def _init_freq(self, n):
        inv_freq = 10000 ** (th.arange(0, self.head_dim, 2).float() / -self.head_dim)
        t = th.arange(n, dtype=th.float32)
        self.freqs = th.outer(t, inv_freq)

    def forward(self, x: Float[Tensor, "B H N D"]) -> Float[Tensor, "B H N D"]:
        if self.freqs.size(0) < x.size(2):
            self._init_freq(x.size(2))
        freqs = self.freqs[None,None,:x.size(2)].to(x.device)
        x1, x2 = x.chunk(2, dim=-1)
        cos = freqs.cos()
        sin = freqs.sin()
        return th.cat([
            x1 * cos - x2 * sin, 
            x1 * sin + x2 * cos,
        ], dim=-1)


class LInfSA(nn.Module):

    def __init__(self, d_model: int, head_dim: int, rope: RoPE, gamma: float = 0.7):
        super().__init__()
        assert d_model % head_dim == 0
        self.head_dim = head_dim
        self.rope = rope
        self.gamma = gamma

        self.qk_proj = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.v_proj = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.out_proj = nn.Conv1d(d_model, d_model, 1, bias=False)

    def forward(self, x: Float[Tensor, "B D L"], eps: float=1e-6, *args, **kwargs) -> Float[Tensor, "B D L"]:
        qk = rearrange(self.qk_proj(x), 'b (h d) n -> b h n d', d=self.head_dim)
        v = rearrange(self.v_proj(x), 'b (h d) n -> b h n d', d=self.head_dim)

        qk = self.rope(qk)

        energies = th.norm(qk, p=2, dim=-1, keepdim=True)
        alpha = energies / (eps + energies.sum(dim=-2, keepdim=True))
        central_query = (alpha * qk).sum(dim=-2, keepdim=True)

        scores = F.relu((qk * central_query).sum(dim=-1, keepdim=True))
        weights = scores / (eps + scores.sum(dim=-2, keepdim=True))

        context = self.gamma * (weights * v).sum(dim=-2, keepdim=True)
        y = context.expand(-1, -1, x.size(-1), -1)

        return self.out_proj(rearrange(y, 'b h n d -> b (h d) n'))


class SDPSA(nn.Module):
    def __init__(self, d_model: int, head_dim: int, rope: RoPE):
        super().__init__()
        assert d_model % head_dim == 0
        self.head_dim = head_dim
        self.rope = rope

        self.qkv_proj = nn.Conv1d(d_model, 3*d_model, 1, bias=False)
        self.out_proj = nn.Conv1d(d_model, d_model, 1, bias=False)

        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)

    def forward(self, x: Float[Tensor, "B D L"], *args, **kwargs) -> Float[Tensor, "B D L"]:
        q,k,v = rearrange(self.qkv_proj(x), 'b (h d) n -> b h n d', d=self.head_dim).chunk(3, dim=1)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = self.rope(q)
        k = self.rope(k)
        y = F.scaled_dot_product_attention(q, k, v)

        # xsa
        vn = F.normalize(v, dim=-1)
        z = y - (y * vn).sum(dim=-1, keepdim=True) * vn

        return self.out_proj(rearrange(z, 'b h n d -> b (h d) n'))