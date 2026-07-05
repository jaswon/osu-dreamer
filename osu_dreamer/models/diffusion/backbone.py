
from dataclasses import dataclass
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.modules.res import Res

from osu_dreamer.modules.attn import SDPSA
from osu_dreamer.modules.swiglu import SwiGLU
from osu_dreamer.modules.rms_norm import RMSNorm

@dataclass
class BackboneArgs:
    depth: int
    expand: int
    head_dim: int
    dropout: float = 0.
    
def resnext(dim: int, expand: int = 1, group_channels: int = 8, radius: int = 1, dilation: int = 1):
    h_dim = dim * expand
    return Res(
        RMSNorm(dim),
        nn.Conv1d(dim, h_dim, 1),
        nn.SiLU(),
        nn.Conv1d(h_dim, h_dim, 1+2*radius, 1, radius*dilation, dilation, groups=h_dim // group_channels),
        nn.SiLU(),
        nn.Conv1d(h_dim, dim, 1),
        RMSNorm(dim),
    )

class Backbone(nn.Module):
    def __init__(
        self,
        dim: int,
        cl_dim: int,
        cg_dim: int,
        args: BackboneArgs,
    ):
        super().__init__()
        self.cond_tower = nn.ModuleList([
            resnext(cl_dim) if cl_dim > 0 else nn.Identity()
            for _ in range(args.depth)
        ])
        
        self.layers = nn.ModuleList([
            nn.ModuleList([
                BackboneLayer(dim, cl_dim, cg_dim, sublayer)
                for sublayer in [
                    SDPSA(dim, args.head_dim),
                    SwiGLU(dim, args.expand, args.dropout, radius=0),
                ]
            ])
            for _ in range(args.depth)
        ])
        self.out_norm = RMSNorm(dim)

    def forward(
        self,
        x: Float[Tensor, "B D L"],
        *,
        cond_l: Float[Tensor, "B Cl L"] | None = None,
        cond_g: Float[Tensor, "B Cg"] | None = None,
    ) -> Float[Tensor, "B D L"]:
        for cond_op, block in zip(self.cond_tower, self.layers):
            if cond_l is not None:
                cond_l = cond_op(cond_l)
            for layer in block: # type: ignore
                x = layer(x, cond_l=cond_l, cond_g=cond_g)
        return self.out_norm(x)

class BackboneLayer(nn.Module):
    def __init__(
        self, 
        dim: int,
        cl_dim: int,
        cg_dim: int, 
        op: nn.Module,
    ):
        super().__init__()
        self.op = op
        self.pre_norm = RMSNorm(dim)
        self.post_norm = RMSNorm(dim, gain=1e-3)
        self.gate = nn.Parameter(th.zeros(dim, 1))

        if cg_dim > 0:
            self.ssg_global = nn.Linear(cg_dim, 3*dim)
            nn.init.zeros_(self.ssg_global.weight)
            nn.init.zeros_(self.ssg_global.bias)

        if cl_dim > 0:
            self.ssg_local = nn.Conv1d(cl_dim, 3*dim, 1)
            nn.init.zeros_(self.ssg_local.weight)
            if self.ssg_local.bias is not None:
                nn.init.zeros_(self.ssg_local.bias)

    def forward(
        self,
        x: Float[Tensor, "B X L"],
        *,
        cond_l: Float[Tensor, "B Cl L"] | None = None,
        cond_g: Float[Tensor, "B Cg"] | None = None,
    ) -> Float[Tensor, "B X L"]:
        if cond_g is None:
            scale_g, shift_g, gate_g = 0, 0, 0
        else:
            scale_g, shift_g, gate_g = self.ssg_global(cond_g)[:,:,None].chunk(3, dim=1)

        if cond_l is None:
            scale_l, shift_l, gate_l = 0, 0, 0
        else:
            scale_l, shift_l, gate_l = self.ssg_local(cond_l).chunk(3, dim=1)

        shift = shift_l + shift_g
        scale = 1 + scale_l + scale_g
        gate = self.gate + gate_l + gate_g

        res = x
        x = self.pre_norm(x) * scale + shift
        x = self.op(x) * gate
        return res + self.post_norm(x)