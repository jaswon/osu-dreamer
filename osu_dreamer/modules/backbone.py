
from dataclasses import dataclass
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.modules.res import Res

from .attn import SDPSA
from .derf import Derf
from .drop_path import DropPath
from .swiglu import SwiGLU

@dataclass
class BackboneArgs:
    depth: int
    expand: int
    head_dim: int
    dropout: float = 0.

class Backbone(nn.Module):
    def __init__(
        self,
        x_dim: int,
        local_cond_dim: int,
        global_cond_dim: int,
        args: BackboneArgs,
    ):
        super().__init__()
        sublayers = [
            lambda: SDPSA(x_dim, args.head_dim), 
            lambda: SwiGLU(x_dim, args.expand, args.dropout, radius=0),
        ]
        self.layers = nn.ModuleList([
            BackboneLayer(x_dim, local_cond_dim, global_cond_dim, args.expand, sublayers[i%len(sublayers)]())
            for i in range(len(sublayers)*args.depth)
        ])
        self.dropouts = nn.ModuleList([
            DropPath(p)
            for p in th.linspace(0., args.dropout, len(sublayers)*args.depth).tolist()
        ])
        self.out_norm = Derf(x_dim, 1)

    def forward(
        self,
        x: Float[Tensor, "B D L"],
        *,
        cond_l: Float[Tensor, "B Cl L"] | None = None,
        cond_g: Float[Tensor, "B Cg"] | None = None,
    ) -> Float[Tensor, "B D L"]:
        for layer, dropout in zip(self.layers, self.dropouts):
            x = x + dropout(layer(x,cond_l=cond_l, cond_g=cond_g))
        return self.out_norm(x)
    

def resnext(dim: int, expand: int, group_channels: int = 8, radius: int = 2, dilation: int = 2):
    h_dim = dim * expand
    return Res(nn.Sequential(
        Derf(dim, 1),
        nn.Conv1d(dim, h_dim, 1),
        nn.SiLU(),
        nn.Conv1d(h_dim, h_dim, 1+2*radius, 1, radius*dilation, dilation, groups=h_dim // group_channels),
        nn.SiLU(),
        nn.Conv1d(h_dim, dim, 1),
    ))

class BackboneLayer(nn.Module):
    def __init__(
        self, 
        dim: int,
        local_cond_dim: int,
        global_cond_dim: int, 
        expand: int,
        op: nn.Module,
    ):
        super().__init__()
        self.op = op
        self.norm = Derf(dim, 1)
        if global_cond_dim > 0:
            self.ssg_global = nn.Linear(global_cond_dim, 3*dim)

        if local_cond_dim > 0:
            self.ssg_local = nn.Sequential(
                resnext(local_cond_dim, expand),
                nn.Conv1d(local_cond_dim, 3*dim, 1),
            )

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
            
        x = self.norm(x) * (1 + scale_l + scale_g) + (shift_l + shift_g)
        x = self.op(x)
        x = x * (1 + gate_l + gate_g)
        return x