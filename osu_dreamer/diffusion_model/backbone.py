
from dataclasses import dataclass
from jaxtyping import Float

from torch import nn, Tensor

from osu_dreamer.modules.res import Res

from osu_dreamer.modules.attn import SDPSA
from osu_dreamer.modules.swiglu import SwiGLU

@dataclass
class BackboneArgs:
    depth: int
    expand: int
    head_dim: int
    dropout: float = 0.
    
def resnext(dim: int, expand: int = 1, group_channels: int = 8, radius: int = 1, dilation: int = 1):
    h_dim = dim * expand
    return Res(nn.Sequential(
        nn.GroupNorm(1, dim),
        nn.Conv1d(dim, h_dim, 1),
        nn.SiLU(),
        nn.Conv1d(h_dim, h_dim, 1+2*radius, 1, radius*dilation, dilation, groups=h_dim // group_channels),
        nn.SiLU(),
        nn.Conv1d(h_dim, dim, 1),
    ))

class Backbone(nn.Module):
    def __init__(
        self,
        dim: int,
        cond_l_dim: int,
        cond_g_dim: int,
        args: BackboneArgs,
    ):
        super().__init__()
        if cond_l_dim > 0:
            self.cond_tower = nn.ModuleList([ resnext(cond_l_dim) for _ in range(args.depth) ])
        self.attns = nn.ModuleList([
            BackboneLayer(dim, cond_l_dim, cond_g_dim, SDPSA(dim, args.head_dim))
            for _ in range(args.depth)
        ])
        self.mlps = nn.ModuleList([
            BackboneLayer(dim, cond_l_dim, cond_g_dim, SwiGLU(dim, args.expand, args.dropout, radius=0))
            for _ in range(args.depth)
        ])
        self.out_norm = nn.GroupNorm(1, dim)

    def forward(
        self,
        x: Float[Tensor, "B D L"],
        *,
        cond_l: Float[Tensor, "B Cl L"] | None = None,
        cond_g: Float[Tensor, "B Cg"] | None = None,
    ) -> Float[Tensor, "B D L"]:
        for i, (attn, mlp) in enumerate(zip(self.attns, self.mlps)):
            if cond_l is not None:
                cond_l = self.cond_tower[i](cond_l)
            x = attn(x, cond_l=cond_l, cond_g=cond_g)
            x = mlp(x, cond_l=cond_l, cond_g=cond_g)
        return self.out_norm(x)

class BackboneLayer(nn.Module):
    def __init__(
        self, 
        dim: int,
        cond_l_dim: int,
        cond_g_dim: int, 
        op: nn.Module,
    ):
        super().__init__()
        self.op = op
        self.pre_norm = nn.GroupNorm(1, dim)
        
        if cond_g_dim > 0:
            self.ssg_global = nn.Linear(cond_g_dim, 3*dim)

        if cond_l_dim > 0:
            self.ssg_local = nn.Conv1d(cond_l_dim, 3*dim, 1)

    def forward(
        self,
        x: Float[Tensor, "B X L"],
        *,
        cond_l: Float[Tensor, "B Cl L"] | None = None,
        cond_g: Float[Tensor, "B Cg"] | None = None,
    ) -> Float[Tensor, "B X L"]:
        res = x
        if cond_g is None:
            scale_g, shift_g, gate_g = 0, 0, 0
        else:
            scale_g, shift_g, gate_g = self.ssg_global(cond_g)[:,:,None].chunk(3, dim=1)

        if cond_l is None:
            scale_l, shift_l, gate_l = 0, 0, 0
        else:
            scale_l, shift_l, gate_l = self.ssg_local(cond_l).chunk(3, dim=1)
            
        x = self.pre_norm(x) * (1 + scale_l + scale_g) + (shift_l + shift_g)
        x = self.op(x)
        x = x * (1 + gate_l + gate_g)
        return res + x