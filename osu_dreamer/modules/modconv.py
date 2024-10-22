
from typing import Callable
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from einops import rearrange

class _ModConv(nn.Module):
    conv: Callable

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        t_dim: int,
    ):
        super().__init__()

        self.weight = nn.Parameter(th.empty(out_dim, in_dim))
        th.nn.init.kaiming_uniform_(self.weight, a=5**.5)

        self.mod = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, in_dim),
        )

    def forward(self, xt: tuple[
        Float[Tensor, "B I ..."], 
        Float[Tensor, "B T"],
    ]) -> Float[Tensor, "B O ..."]:
        bx,t = xt
        b = t.size(0)

        bw = th.einsum('oi,bi->boi', self.weight, self.mod(t)+1)
        demod = th.rsqrt(th.sum(bw ** 2, dim=2) + 1e-8)
        bw = th.einsum('boi,bo->boi', bw, demod)

        w = rearrange(bw, 'b o i -> (b o) i')
        x = rearrange(bx, 'b d ... -> 1 (b d) ...') 
        o = type(self).conv(x, w, b)

        return rearrange(o, '1 (b d) ... -> b d ...', b=b)

class ModulatedConv1d(_ModConv):
    conv = lambda x,w,b: F.conv1d(x, w[:,:,None], None, groups=b)

class ModulatedConv2d(_ModConv):
    conv = lambda x,w,b: F.conv2d(x, w[:,:,None,None], None, groups=b)