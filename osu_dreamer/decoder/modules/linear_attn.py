
from jaxtyping import Float

import torch as th
from torch import Tensor

from einops import pack, unpack


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