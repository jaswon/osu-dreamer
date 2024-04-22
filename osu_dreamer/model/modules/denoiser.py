
from dataclasses import dataclass

from jaxtyping import Float, Int

import torch as th
from torch import nn, Tensor

from .residual import ResiDual
from .s4d import S4Block
from .scaleshift import ScaleShift
    
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

@dataclass
class EncoderArgs:
    h_dim: int
    num_layers: int
    expand: int

class Encoder(nn.Sequential):
    def __init__(self, a_dim: int, args: EncoderArgs):
        super().__init__(
            nn.Conv1d(a_dim, args.h_dim, 1),
            ResiDual(args.h_dim, [
                S4Block(args.h_dim, args.expand)
                for _ in range(args.num_layers)
            ]),
        )

@dataclass
class DenoiserArgs:
    t_dim: int
    proj_dim: int
    h_dim: int
    num_layers: int
    expand: int

class Denoiser(nn.Module):
    def __init__(
        self,
        x_dim: int,
        a_dim: int,
        args: DenoiserArgs,
    ):
        super().__init__()

        self.proj_t = nn.Sequential(
            GaussianFourierProjection(args.t_dim),
            nn.Linear(args.t_dim, args.t_dim),
            nn.LayerNorm(args.t_dim),
            nn.SiLU(),
            nn.Linear(args.t_dim, args.t_dim),
            nn.LayerNorm(args.t_dim),
            nn.SiLU(),
        )

        in_dim = a_dim + x_dim + x_dim
        self.proj_in = nn.Sequential(
            nn.Conv1d(in_dim, args.proj_dim, 1),
            nn.GroupNorm(1, args.proj_dim),
            nn.SiLU(),
            nn.Conv1d(args.proj_dim, args.h_dim, 1),
        )

        self.net = ResiDual(args.h_dim, [
            ScaleShift(args.h_dim, args.t_dim, S4Block(args.h_dim, args.expand))
            for _ in range(args.num_layers)
        ])
        
        self.proj_out = nn.Conv1d(args.h_dim, x_dim, 1)
        self.proj_out.weight.data.fill_(0.)
        self.proj_out.bias.data.fill_(0.) # type: ignore

    def forward(
        self, 
        a: Float[Tensor, "B A L"],
        p: Int[Tensor, "B L"],
        y: Float[Tensor, "B X L"],
        x: Float[Tensor, "B X L"],
        t: Float[Tensor, "B"],
    ) -> Float[Tensor, "B X L"]:
        t = self.proj_t(t)
        h = self.proj_in(th.cat([a, x, y], dim=1))
        o = self.net(h, t)
        return self.proj_out(o)