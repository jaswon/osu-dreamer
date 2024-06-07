
from dataclasses import dataclass

from jaxtyping import Float, Int

import torch as th
from torch import nn, Tensor

from osu_dreamer.common.residual import ResStack
    
from .scaleshift import ScaleShift
from .encoder import Encoder, EncoderArgs
    
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
class DenoiserArgs:
    t_features: int
    t_dim: int
    h_dim: int
    mlp_depth: int
    encoder_args: EncoderArgs

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

        in_dim = a_dim + x_dim + x_dim
        self.proj_in = nn.Conv1d(in_dim, args.h_dim, 1)

        self.mlp = ResStack(args.h_dim, [
            ScaleShift(args.h_dim, args.t_dim, nn.Sequential(
                nn.SiLU(),
                nn.Conv1d(args.h_dim, args.h_dim, 5,1,2, groups=args.h_dim),
            ))
            for _ in range(args.mlp_depth)
        ])

        self.encoder = Encoder(args.h_dim, args.encoder_args)
        
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
        h = self.proj_in(th.cat([a, x, y], dim=1))
        h = self.mlp(h, t)
        o = self.encoder(h)
        return self.proj_out(o)