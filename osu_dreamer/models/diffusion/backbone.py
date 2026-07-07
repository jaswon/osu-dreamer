
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
            BackboneLayer(dim, cl_dim, cg_dim, sublayer)
            for _ in range(args.depth)
            for sublayer in [
                SDPSA(dim, args.head_dim),
                SwiGLU(dim, args.expand, args.dropout, args.radius),
            ]
        ])
        self.out_norm = RMSNorm(dim)

    def forward(
        self,
        x: Float[Tensor, "B D L"],
        *,
        cl: Float[Tensor, "#B Cl L"] | None = None,
        cg: Float[Tensor, "#B Cg"] | None = None,
    ) -> Float[Tensor, "B D L"]:
        for layer in self.layers:
            x = layer(x, cl=cl, cg=cg)
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
        self.post_norm = RMSNorm(dim, affine=False)
        self.gate = nn.Parameter(th.zeros(dim, 1))

        if cl_dim > 0:
            self.l_ssg = nn.Conv1d(cl_dim, 3*dim, 1)
            nn.init.zeros_(self.l_ssg.weight)
            if self.l_ssg.bias is not None:
                nn.init.zeros_(self.l_ssg.bias)

        if cg_dim > 0:
            self.g_ssg = nn.Linear(cg_dim, 3*dim)
            nn.init.zeros_(self.g_ssg.weight)
            nn.init.zeros_(self.g_ssg.bias)

    def forward(
        self,
        x: Float[Tensor, "B X L"],
        *,
        cl: Float[Tensor, "#B Cl L"] | None = None,
        cg: Float[Tensor, "#B Cg"] | None = None,
    ) -> Float[Tensor, "B X L"]:
        if cl is None:
            l_scale, l_shift, l_gate = 0, 0, 0
        else:
            l_scale, l_shift, l_gate = self.l_ssg(cl).chunk(3, dim=1)
            
        if cg is None:
            g_scale, g_shift, g_gate = 0, 0, 0
        else:
            g_scale, g_shift, g_gate = self.g_ssg(cg)[:,:,None].chunk(3, dim=1)

        res = x
        x = self.pre_norm(x) * (1 + g_scale + l_scale) + (g_shift + l_shift)
        x = self.op(x)
        x = self.post_norm(x) * (self.gate + g_gate + l_gate)
        return res + x