
from typing import Optional
from jaxtyping import Float

from contextlib import contextmanager

import torch as th
from torch import nn, Tensor

from einops import rearrange

class ModulateConv:
    value: Optional[Tensor]

    def __init__(self, t_dim: int, depth: int):
        self.t_dim = t_dim
        self.depth = depth

    @contextmanager
    def set(self, value: Float[Tensor, "B {self.t_dim}"]):
        self.value = value
        yield
        self.value = None

    def __call__(self, conv: nn.modules.conv._ConvNd) -> nn.Module:
        return _ModConv(self, conv)

class _ModConv(nn.Module):
    def __init__(
        self,
        mod: ModulateConv,
        conv: nn.modules.conv._ConvNd,
    ):
        super().__init__()

        self.conv = conv
        self.groups = self.conv.groups

        self.mod = mod
        t_dim = mod.t_dim
        self.proj_t = nn.Sequential(
            *(
                block for _ in range(mod.depth)
                for block in [ nn.Linear(t_dim, t_dim), nn.SiLU() ]
            ),
            nn.Linear(t_dim, self.conv.weight.size(1)),
        )

    def forward(self, bx: Float[Tensor, "B I ..."]) -> Float[Tensor, "B O ..."]:
        t = self.mod.value
        if t is None:
            raise RuntimeError('mod not set')
        B = t.size(0)

        bw = th.einsum('oi...,bi->boi...', self.conv.weight, self.proj_t(t)+1)
        demod = th.rsqrt(th.sum(bw.flatten(2) ** 2, dim=2) + 1e-8)
        bw = th.einsum('boi...,bo->boi...', bw, demod)
        w = rearrange(bw, 'b o i ... -> (b o) i ...')

        x = rearrange(bx, 'b d ... -> 1 (b d) ...') 
        self.conv.groups = B * self.groups
        o = self.conv._conv_forward(x, w, None)
        bo = rearrange(o, '1 (b d) ... -> b d ...', b=B)

        if self.conv.bias is not None:
            bo = bo + self.conv.bias[None,:,None]

        return bo