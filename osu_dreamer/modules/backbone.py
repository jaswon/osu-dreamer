
from dataclasses import dataclass
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

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
            BackboneLayer(x_dim, local_cond_dim, global_cond_dim, sublayers[i%len(sublayers)]())
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
    

class BackboneLayer(nn.Module):
    def __init__(
        self, 
        x_dim: int,
        local_cond_dim: int,
        global_cond_dim: int, 
        op: nn.Module,
    ):
        super().__init__()
        self.op = op
        self.norm = Derf(x_dim, 1)
        if global_cond_dim > 0:
            self.ss_global = nn.Linear(global_cond_dim, 2*x_dim)

        if local_cond_dim > 0:
            self.ss_local = nn.Conv1d(local_cond_dim, 2*x_dim, 1)
        self.alpha = nn.Parameter(th.randn(x_dim, 1))

    def forward(
        self,
        x: Float[Tensor, "B X L"],
        *,
        cond_l: Float[Tensor, "B Cl L"] | None = None,
        cond_g: Float[Tensor, "B Cg"] | None = None,
    ) -> Float[Tensor, "B X L"]:
        if cond_g is None:
            scale_global, shift_global = 0, 0
        else:
            scale_global, shift_global = self.ss_global(cond_g)[:,:,None].chunk(2, dim=1)

        if cond_l is None:
            scale_local, shift_local = 0, 0
        else:
            scale_local, shift_local = self.ss_local(cond_l).chunk(2, dim=1)
            
        x = self.norm(x) * (1 + scale_local + scale_global) + (shift_local + shift_global)
        x = self.op(x)
        x = x * self.alpha
        return x