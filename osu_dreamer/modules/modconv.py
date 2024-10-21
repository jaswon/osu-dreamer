
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from einops import rearrange

def mod_conv(
    conv: nn.modules.conv._ConvNd,
    mod: nn.Linear,
    xt: tuple[
        Float[Tensor, "B I ..."], 
        Float[Tensor, "B T"],
    ],
) -> Float[Tensor, "B O ..."]:
    bx,t = xt
    b = t.size(0)

    bw = th.einsum('oi...,bi->boi...', conv.weight, mod(t)+1)
    demod = th.rsqrt(th.sum(bw.flatten(2) ** 2, dim=-1) + 1e-8)
    bw = th.einsum('boi...,bo->boi...', bw, demod)

    w = rearrange(bw, 'b o i ... -> (b o) i ...')
    x = rearrange(bx, 'b d ... -> 1 (b d) ...')
    conv.groups = b
    o = conv._conv_forward(x, w, None)
    return rearrange(o, '1 (b d) ... -> b d ...', b=b)


class ModulatedConv1d(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        t_dim: int,
    ):
        super().__init__()
        self.proj = nn.Conv1d(in_dim, out_dim, 1, bias=False)
        self.mod = nn.Linear(t_dim, in_dim)

    def forward(self, xt) -> Float[Tensor, "B O L"]:
        return mod_conv(self.proj, self.mod, xt)

class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        t_dim: int,
    ):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, 1, bias=False)
        self.mod = nn.Linear(t_dim, in_dim)

    def forward(self, xt) -> Float[Tensor, "B O W H"]:
        return mod_conv(self.proj, self.mod, xt)