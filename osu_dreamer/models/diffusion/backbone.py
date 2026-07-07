
from dataclasses import dataclass
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.modules.attn import SDPSA
from osu_dreamer.modules.swiglu import SwiGLU
from osu_dreamer.modules.rms_norm import RMSNorm

@dataclass
class BackboneArgs:
    depth: int
    expand: int
    head_dim: int
    dropout: float = 0.

class Backbone(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_dim: int,
        args: BackboneArgs,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            BackboneLayer(dim, cond_dim, sublayer)
            for _ in range(args.depth)
            for sublayer in [
                SDPSA(dim, args.head_dim),
                SwiGLU(dim, args.expand, args.dropout, radius=0),
            ]
        ])
        self.out_norm = RMSNorm(dim)

    def forward(
        self,
        x: Float[Tensor, "B D L"],
        *,
        cond: Float[Tensor, "B C"] | None = None,
    ) -> Float[Tensor, "B D L"]:
        for layer in self.layers:
            x = layer(x, cond=cond)
        return self.out_norm(x)

class BackboneLayer(nn.Module):
    def __init__(
        self, 
        dim: int,
        cg_dim: int, 
        op: nn.Module,
    ):
        super().__init__()
        self.op = op
        self.pre_norm = RMSNorm(dim)
        self.post_norm = RMSNorm(dim, affine=False)
        self.gate = nn.Parameter(th.zeros(dim, 1))

        if cg_dim > 0:
            self.ssg = nn.Linear(cg_dim, 3*dim)
            nn.init.zeros_(self.ssg.weight)
            nn.init.zeros_(self.ssg.bias)

    def forward(
        self,
        x: Float[Tensor, "B X L"],
        *,
        cond: Float[Tensor, "B C"] | None = None,
    ) -> Float[Tensor, "B X L"]:
        if cond is None:
            scale, shift, gate = 0, 0, 0
        else:
            scale, shift, gate = self.ssg(cond)[:,:,None].chunk(3, dim=1)

        res = x
        x = self.pre_norm(x) * (1 + scale) + shift
        x = self.op(x)
        x = self.post_norm(x) * (self.gate + gate)
        return res + x