
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.data.prepare_map import NUM_LABELS

from .residual import ResStack
from .unet import UNet
from .cbam import CBAM
    
class RandomFourierFeatures(nn.Module):
    def __init__(self, in_dim, out_dim, scale=30):
        super().__init__()
        d = out_dim // 2
        assert d*2 == out_dim, '`out_dim` must be even'
        self.W = nn.Parameter(th.randn(in_dim, d) * scale, requires_grad=False)

    def forward(self, x: Float[Tensor, "... D"]) -> Float[Tensor, "... E"]:
        theta = x @ self.W * 2 * th.pi
        return th.cat([theta.sin(), theta.cos()], dim=-1)

class ScaleShift(nn.Module):
    def __init__(self, dim: int, t_dim: int, net: nn.Module):
        super().__init__()
        self.net = net
        self.act = nn.Sequential(
            nn.Conv1d(dim, 2*dim, 1),
            nn.GLU(dim=1),
        )

        self.norm = nn.GroupNorm(dim, dim, affine=False)
        self.ss = nn.Linear(t_dim, dim*2)
        nn.init.zeros_(self.ss.weight)
        nn.init.zeros_(self.ss.bias)

    def forward(self, x: Float[Tensor, "B X L"], t: Float[Tensor, "B T"]):
        scale, shift = self.ss(t)[...,None].chunk(2, dim=1)
        o = self.net(self.norm(x) * (1+scale) + shift)
        return self.act(o)

@dataclass
class DenoiserArgs:
    rff_dim: int
    ss_dim: int
    h_dim: int
    scales: list[int]
    block_depth: int
    stack_depth: int

class Denoiser(nn.Module):
    def __init__(
        self,
        x_dim: int,
        a_dim: int,
        args: DenoiserArgs,
    ):
        super().__init__()

        self.proj_cond = nn.Sequential(
            RandomFourierFeatures(2 + NUM_LABELS, args.rff_dim),
            nn.Linear(args.rff_dim, args.ss_dim),
            nn.SiLU(),
        )

        self.proj_h = nn.Conv1d(a_dim+x_dim+x_dim, args.h_dim, 1)
        
        self.net = UNet(
            args.h_dim, args.scales,
            ResStack(args.h_dim, [
                ScaleShift(args.h_dim, args.ss_dim, block)
                for _ in range(args.stack_depth)
                for block in [
                    nn.Conv1d(args.h_dim, args.h_dim, 5,1,2, groups=args.h_dim),
                    CBAM(args.h_dim),
                ]
            ]),
            lambda: ResStack(args.h_dim, [
                ScaleShift(args.h_dim, args.ss_dim, nn.Conv1d(args.h_dim, args.h_dim, 3,1,1, groups=args.h_dim))
                for _ in range(args.block_depth)
            ]),
        )

        self.proj_out = nn.Conv1d(args.h_dim, x_dim, 1)
        th.nn.init.zeros_(self.proj_out.weight)
        th.nn.init.zeros_(self.proj_out.bias) # type: ignore

    def forward(
        self, 
        a: Float[Tensor, "B A L"],
        star_rating: Float[Tensor, "B 1"],
        diff_labels: Float[Tensor, str(f"B {NUM_LABELS}")],
        # --- diffusion args --- #
        y: Float[Tensor, "B X L"],
        x: Float[Tensor, "B X L"],
        t: Float[Tensor, "B"],
    ) -> Float[Tensor, "B X L"]:
        c = self.proj_cond(th.cat([t[:,None],star_rating,diff_labels], dim=1))
        h = self.proj_h(th.cat([a,x,y], dim=1))
        return self.proj_out(self.net(h,c))