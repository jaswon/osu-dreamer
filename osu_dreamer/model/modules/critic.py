
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from .wavenet import WaveNet, WaveNetArgs

@dataclass
class CriticArgs:
    h_dim: int
    a_pre_args: WaveNetArgs
    scales: list[int]

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
                    nn.Conv1d(args.h_dim, args.h_dim, scale, scale),
                    nn.BatchNorm1d(args.h_dim),
                    nn.SiLU(),
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
        x: Float[Tensor, "B X L"],
    ) -> Float[Tensor, "B l"]:
        return self.net(th.cat([self.a_pre(a),x], dim=1)).squeeze(1)