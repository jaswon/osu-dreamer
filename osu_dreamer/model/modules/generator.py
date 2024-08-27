
from dataclasses import dataclass
from typing import Optional

from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from .residual import ResStack
from .wavenet import WaveNet, WaveNetArgs
from .unet import UNet
from .linear_attn import RoPE, LinearAttn, AttnArgs

class ScaleShift(nn.Module):
    def __init__(self, dim: int, t_dim: int, net: nn.Module):
        super().__init__()
        self.net = net

        self.ss = nn.Linear(t_dim, dim*2)
        nn.init.zeros_(self.ss.weight)
        nn.init.zeros_(self.ss.bias)

    def forward(self, x: Float[Tensor, "B X L"], t: Float[Tensor, "B T"]):
        scale, shift = self.ss(t)[...,None].chunk(2, dim=1)
        return self.net(x * (1+scale) + shift)

@dataclass
class GeneratorArgs:
    z_dim: int
    z_h_dim: int
    h_dim: int
    a_pre_args: WaveNetArgs

    scales: list[int]
    block_depth: int
    stack_depth: int
    attn_args: AttnArgs

class Generator(nn.Module):
    def __init__(
        self,
        x_dim: int,
        a_dim: int,
        args: GeneratorArgs,
    ):
        super().__init__()
        self.z_dim = args.z_dim

        self.proj_z = nn.Sequential(
            nn.Linear(args.z_dim, args.z_h_dim),
            nn.LayerNorm(args.z_h_dim),
            nn.SiLU(),
            nn.Linear(args.z_h_dim, args.z_h_dim),
            nn.LayerNorm(args.z_h_dim),
            nn.SiLU(),
        )

        self.proj_in = nn.Sequential(
            nn.Conv1d(a_dim, args.h_dim, 1),
            WaveNet(args.h_dim, args.a_pre_args),
        )
        
        self.rope = RoPE(args.attn_args.head_dim)
        self.net = UNet(
            args.h_dim, args.scales,
            ResStack(args.h_dim, [
                ScaleShift(args.h_dim, args.z_h_dim, block)
                for _ in range(args.stack_depth)
                for block in [
                    LinearAttn(args.h_dim, self.rope, args.attn_args),
                    nn.Conv1d(args.h_dim, args.h_dim, 3,1,1, groups=args.h_dim),
                ]
            ]),
            lambda: ResStack(args.h_dim, [
                ScaleShift(args.h_dim, args.z_h_dim, nn.Conv1d(args.h_dim, args.h_dim, 3,1,1, groups=args.h_dim))
                for _ in range(args.block_depth)
            ]),
        )

        self.proj_out = nn.Conv1d(args.h_dim, x_dim, 1)
        th.nn.init.zeros_(self.proj_out.weight)
        th.nn.init.zeros_(self.proj_out.bias) # type: ignore

    def forward(
        self, 
        a: Float[Tensor, "B A L"],
        z: Optional[Float[Tensor, "B Z"]] = None,
    ) -> Float[Tensor, "B X L"]:
        if z is None:
            z = th.randn(a.size(0), self.z_dim, device=a.device)
        
        h = self.proj_in(a)
        o = self.net(h,self.proj_z(z))
        return self.proj_out(o).clamp(min=-1, max=1)