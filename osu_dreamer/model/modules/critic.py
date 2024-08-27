
from dataclasses import dataclass

from jaxtyping import Float, Int

import torch as th
from torch import nn, Tensor

from .residual import ResStack, WaveNet, WaveNetArgs

@dataclass
class CriticArgs:
    h_dim: int
    a_pre_args: WaveNetArgs
    scales: list[int]
    block_depth: int

class Critic(nn.Module):
    def __init__(
        self,
        x_dim: int,
        a_dim: int,
        args: CriticArgs,
    ):
        super().__init__()

        self.a_pre = nn.Sequential(
            nn.Conv1d(a_dim, args.h_dim, 1),
            WaveNet(args.h_dim, args.a_pre_args),
        )

        self.net = nn.Sequential(
            nn.Conv1d(x_dim+args.h_dim, args.h_dim, 1), 
            *(
                block for scale in args.scales
                for block in [
                    ResStack(args.h_dim, [
                        nn.Conv1d(args.h_dim, args.h_dim, 1)
                        for _ in range(args.block_depth)
                    ]),
                    nn.Conv1d(args.h_dim, args.h_dim, scale, scale),
                ]
            ),
            nn.Conv1d(args.h_dim, 1, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.utils.parametrizations.spectral_norm(m)

    def forward(
        self,
        a: Float[Tensor, "B A L"],
        p: Int[Tensor, "B L"],
        x: Float[Tensor, "B X L"],
    ) -> Float[Tensor, "B l"]:
        return self.net(th.cat([self.a_pre(a),x], dim=1)).squeeze(1)