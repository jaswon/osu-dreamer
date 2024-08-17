
from dataclasses import dataclass

from jaxtyping import Float, Int

import torch as th
from torch import nn, Tensor

from osu_dreamer.common.norm import RMSNorm
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
    def __init__(self, dim: int, cond_dim: int, net: nn.Module):
        super().__init__()
        self.net = net

        self.to_scale_shift = nn.Linear(cond_dim, dim * 2)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: Float[Tensor, "B D L"], e: Float[Tensor, "B T"]):
        scale, shift = self.to_scale_shift(e).unsqueeze(-1).chunk(2, dim=1)
        return self.net(x * (1+scale) + shift)

@dataclass
class DenoiserArgs:
    t_features: int
    t_dim: int
    h_dim: int
    scales: list[int]
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

        self.proj_in = nn.Conv1d(a_dim + x_dim + x_dim, args.h_dim, 1)
        
        self.rope = RoPE(args.attn_args.head_dim)
        self.net = ResStack(args.h_dim, [
            ScaleShift(args.h_dim, args.t_dim, block)
            for _ in range(args.stack_depth)
            for block in [
                UNet(
                    args.h_dim, args.scales, 
                    LinearAttn(args.h_dim, self.rope, args.attn_args), 
                    lambda: nn.Sequential(
                        nn.Conv1d(args.h_dim, args.h_dim * 2, 5,1,2, groups=args.h_dim),
                        nn.Conv1d(args.h_dim * 2, args.h_dim, 1),
                    ),
                ),
                nn.Conv1d(args.h_dim, args.h_dim, 5,1,2, groups=args.h_dim),
            ]
        ])

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
        h = self.proj_in(th.cat([a, x, y], dim=1))
        o = self.net(h, self.proj_t(t))
        return self.proj_out(o)