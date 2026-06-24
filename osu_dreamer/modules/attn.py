
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from einops import rearrange

_rope_cache: dict[tuple[int, th.device], Float[Tensor, "N d"]] = {}

def rope(x: Float[Tensor, "B H N D"]) -> Float[Tensor, "B H N D"]:
    _,_,N,D = x.size()
    assert D % 2 == 0, "head_dim must be even"

    key = (D, x.device)
    if key not in _rope_cache or _rope_cache[key].size(0) < N:
        inv_freq = 10000 ** (th.arange(0, D, 2).float() / -D)
        t = th.arange(N, dtype=th.float32)
        _rope_cache[key] = th.outer(t, inv_freq).to(x.device)
    freqs = _rope_cache[key][:x.size(2)]

    x1, x2 = x.chunk(2, dim=-1)
    cos = freqs.cos().to(x.dtype)
    sin = freqs.sin().to(x.dtype)
    return th.cat([
        x1 * cos - x2 * sin, 
        x1 * sin + x2 * cos,
    ], dim=-1)

class LInfSA(nn.Module):

    def __init__(self, d_model: int, head_dim: int, gamma: float = 0.7):
        super().__init__()
        assert d_model % head_dim == 0
        self.head_dim = head_dim
        self.gamma = gamma

        self.qk_proj = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.v_proj = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.out_proj = nn.Conv1d(d_model, d_model, 1, bias=False)

    def forward(self, x: Float[Tensor, "B D L"], eps: float=1e-6, *args, **kwargs) -> Float[Tensor, "B D L"]:
        qk = rearrange(self.qk_proj(x), 'b (h d) n -> b h n d', d=self.head_dim)
        v = rearrange(self.v_proj(x), 'b (h d) n -> b h n d', d=self.head_dim)

        qk = rope(qk)

        energies = th.norm(qk, p=2, dim=-1, keepdim=True)
        alpha = energies / (eps + energies.sum(dim=-2, keepdim=True))
        central_query = (alpha * qk).sum(dim=-2, keepdim=True)

        scores = F.relu((qk * central_query).sum(dim=-1, keepdim=True))
        weights = scores / (eps + scores.sum(dim=-2, keepdim=True))

        context = self.gamma * (weights * v).sum(dim=-2, keepdim=True)
        y = context.expand(-1, -1, x.size(-1), -1)

        return self.out_proj(rearrange(y, 'b h n d -> b (h d) n'))


class SDPSA(nn.Module):
    def __init__(self, d_model: int, head_dim: int):
        super().__init__()
        assert d_model % head_dim == 0
        self.head_dim = head_dim

        self.qkv_proj = nn.Conv1d(d_model, 3*d_model, 1, bias=False)
        self.out_proj = nn.Conv1d(d_model, d_model, 1, bias=False)

        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)

    def forward(self, x: Float[Tensor, "B D L"], *args, **kwargs) -> Float[Tensor, "B D L"]:
        q,k,v = rearrange(self.qkv_proj(x), 'b (h d) n -> b h n d', d=self.head_dim).chunk(3, dim=1)

        q = self.q_norm(q.float()).type_as(q)
        k = self.k_norm(k.float()).type_as(k)

        q = rope(q)
        k = rope(k)
        y = F.scaled_dot_product_attention(q, k, v)

        # xsa
        vn = F.normalize(v, dim=-1)
        z = y - (y * vn).sum(dim=-1, keepdim=True) * vn

        return self.out_proj(rearrange(z, 'b h n d -> b (h d) n'))