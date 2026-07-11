
from dataclasses import dataclass
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.common.attn import SDPSA
from osu_dreamer.common.swiglu import SwiGLU
from osu_dreamer.common.rms_norm import RMSNorm

def zero(m: nn.Linear | nn.Conv1d):
    nn.init.zeros_(m.weight)
    if m.bias is not None:
        nn.init.zeros_(m.bias)
    return m

@dataclass
class BackboneArgs:
    depth: int
    expand: int
    head_dim: int
    n_heads: int
    radius: int = 1
    dropout: float = 0.

class Backbone(nn.Module):
    def __init__(
        self,
        dim: int,
        cl_dim: int,
        cg_dim: int,
        args: BackboneArgs,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            BackboneLayer(dim, cl_dim, cg_dim, args)
            for _ in range(args.depth)
        ])
        self.out_norm = RMSNorm(dim)

    def forward(
        self,
        x: Float[Tensor, "B D L"],
        cl: Float[Tensor, "#B Cl L"],
        cg: Float[Tensor, "#B Cg"],
    ) -> Float[Tensor, "B D L"]:
        for layer in self.layers:
            x = layer(x, cl, cg)
        return self.out_norm(x)

class BackboneLayer(nn.Module):
    def __init__(
        self, 
        dim: int,
        cl_dim: int,
        cg_dim: int, 
        args: BackboneArgs,
    ):
        super().__init__()
        self.norm = RMSNorm(dim, affine=False)

        self.ssg1 = zero(nn.Linear(cg_dim, 3*dim))
        self.attn = SDPSA(dim+cl_dim, args.n_heads, args.head_dim, d_out=dim)

        self.ssg2 = zero(nn.Linear(cg_dim, 3*dim))
        self.ffn = SwiGLU(dim, args.expand, args.dropout, args.radius)

    def forward(
        self,
        x: Float[Tensor, "B X L"],
        cl: Float[Tensor, "#B Cl L"],
        cg: Float[Tensor, "#B Cg"],
    ) -> Float[Tensor, "B X L"]:
        
        scale, shift, gate = self.ssg1(cg)[:,:,None].chunk(3, dim=1)
        h = self.norm(x) * (1 + scale) + shift
        h = self.attn(th.cat([cl.expand(x.size(0),-1,-1), h], dim=1)) * gate
        x = x + h

        scale, shift, gate = self.ssg2(cg)[:,:,None].chunk(3, dim=1)
        h = self.norm(x) * (1 + scale) + shift
        h = self.ffn(h) * gate
        x = x + h

        return x