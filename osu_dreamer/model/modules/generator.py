
from dataclasses import dataclass
from typing import Optional

from jaxtyping import Float, Int

import torch as th
from torch import nn, Tensor

from osu_dreamer.common.residual import ResStack
from osu_dreamer.common.unet import UNet
from osu_dreamer.common.linear_attn import RoPE, LinearAttn, AttnArgs

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
    h_dim: int

    enc_stack_depth: int
    scales: list[int]
    block_depth: int
    stack_depth: int
    attn_args: AttnArgs

class Generator(nn.Module):
    def __init__(
        self,
        z_dim: int,
        x_dim: int,
        a_dim: int,
        args: GeneratorArgs,
    ):
        super().__init__()
        self.z_dim = z_dim

        self.proj_in = nn.Sequential(
            nn.Conv1d(a_dim, args.h_dim, 1), 
            ResStack(args.h_dim, [
                nn.Conv1d(args.h_dim, args.h_dim, 5,1,2, groups=args.h_dim)
                for _ in range(args.enc_stack_depth)
            ]),
        )
        
        self.rope = RoPE(args.attn_args.head_dim)
        self.net = UNet(
            args.h_dim, args.scales,
            ResStack(args.h_dim, [
                ScaleShift(args.h_dim, z_dim, block)
                for _ in range(args.stack_depth)
                for block in [
                    LinearAttn(args.h_dim, self.rope, args.attn_args),
                    nn.Conv1d(args.h_dim, args.h_dim, 5,1,2, groups=args.h_dim),
                ]
            ]),
            lambda: ResStack(args.h_dim, [
                nn.Conv1d(args.h_dim, args.h_dim, 5,1,2, groups=args.h_dim)
                for _ in range(args.block_depth)
            ]),
        )

        self.proj_out = nn.Conv1d(args.h_dim, x_dim, 1)

    def forward(
        self, 
        a: Float[Tensor, "B A L"],
        p: Int[Tensor, "B L"],
        seed: Optional[int] = None,
    ) -> Float[Tensor, "B X L"]:
        rng = None
        if seed is not None:
            rng = th.Generator()
            rng.manual_seed(seed)
        z = th.randn(a.size(0), self.z_dim, generator=rng, device=a.device)
        
        h = self.proj_in(a)
        o = self.net(h,z)
        return self.proj_out(o).clamp(min=-1, max=1)