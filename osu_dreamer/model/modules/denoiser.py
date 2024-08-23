
from dataclasses import dataclass

from jaxtyping import Float, Int

import torch as th
from torch import nn, Tensor

from osu_dreamer.common.residual import ResStack
from osu_dreamer.common.unet import UNet
from osu_dreamer.common.linear_attn import RoPE, LinearAttn, AttnArgs
    
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""  
    def __init__(self, dim, scale=30.):
        super().__init__()
        d = dim // 2
        assert d*2 == dim, '`dim` must be even'
        self.W = nn.Parameter(th.randn(d) * scale, requires_grad=False)

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "... E"]:
        theta = x[:, None] * self.W[None, :] * 2 * th.pi
        return th.cat([theta.sin(), theta.cos()], dim=-1)

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
class DenoiserArgs:
    t_features: int
    t_dim: int
    h_dim: int
    scales: list[int]
    block_depth: int
    stack_depth: int
    attn_args: AttnArgs

class Denoiser(nn.Module):
    def __init__(
        self,
        x_dim: int,
        a_dim: int,
        args: DenoiserArgs,
    ):
        super().__init__()

        self.proj_t = nn.Sequential(
            GaussianFourierProjection(args.t_features * 2),
            nn.Linear(args.t_features * 2, args.t_dim),
            nn.LayerNorm(args.t_dim),
            nn.SiLU(),
            nn.Linear(args.t_dim, args.t_dim),
            nn.LayerNorm(args.t_dim),
            nn.SiLU(),
        )

        self.proj_h = nn.Conv1d(a_dim+x_dim+x_dim, args.h_dim, 1)
        
        self.rope = RoPE(args.attn_args.head_dim)
        self.net = UNet(
            args.h_dim, args.scales,
            ResStack(args.h_dim, [
                ScaleShift(args.h_dim, args.t_dim, block)
                for _ in range(args.stack_depth)
                for block in [
                    LinearAttn(args.h_dim, self.rope, args.attn_args),
                    nn.Conv1d(args.h_dim, args.h_dim, 3,1,1, groups=args.h_dim),
                ]
            ]),
            lambda: ResStack(args.h_dim, [
                ScaleShift(args.h_dim, args.t_dim, nn.Conv1d(args.h_dim, args.h_dim, 3,1,1, groups=args.h_dim))
                for _ in range(args.block_depth)
            ]),
        )

        self.proj_out = nn.Conv1d(args.h_dim, x_dim, 1)
        th.nn.init.zeros_(self.proj_out.weight)
        th.nn.init.zeros_(self.proj_out.bias) # type: ignore

    def forward(
        self, 
        a: Float[Tensor, "B A L"],
        p: Int[Tensor, "B L"],
        y: Float[Tensor, "B X L"],
        x: Float[Tensor, "B X L"],
        t: Float[Tensor, "B"],
    ) -> Float[Tensor, "B X L"]:
        t = self.proj_t(t)
        h = self.proj_h(th.cat([a,x,y], dim=1))
        return self.proj_out(self.net(h,t))