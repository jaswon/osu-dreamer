
from dataclasses import dataclass

from jaxtyping import Float, Int

import torch as th
from torch import nn, Tensor

from osu_dreamer.common.residual import ResStack


@dataclass
class CriticArgs:
    h_dim: int
    a_pre_num_blocks: int
    a_pre_block_depth: int
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
            ResStack(args.h_dim, [
                nn.Sequential(
                    nn.ZeroPad1d((1,0) if d==0 else 2**(d-1)),
                    nn.Conv1d(
                        args.h_dim, args.h_dim, 2, 
                        dilation=2**d,
                        groups=args.h_dim,
                    )
                )
                for _ in range(args.a_pre_num_blocks)
                for d in range(args.a_pre_block_depth)
            ]) # wave net: context length = num_blocks*2^block_depth
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

    def forward(
        self,
        a: Float[Tensor, "B A L"],
        p: Int[Tensor, "B L"],
        x: Float[Tensor, "B X L"],
    ) -> Float[Tensor, "B l"]:
        return self.net(th.cat([self.a_pre(a),x], dim=1)).squeeze(1)