
from dataclasses import dataclass

from jaxtyping import Float, Int

import torch as th
from torch import nn, Tensor

from osu_dreamer.common.residual import ResStack
from osu_dreamer.common.linear_attn import RoPE, LinearAttn, AttnArgs



@dataclass
class CriticArgs:
    h_dim: int

    scales: list[int]
    block_depth: int
    stack_depth: int
    attn_args: AttnArgs

class Critic(nn.Module):
    def __init__(
        self,
        x_dim: int,
        a_dim: int,
        args: CriticArgs,
    ):
        super().__init__()

        self.rope = RoPE(args.attn_args.head_dim)
        self.net = nn.Sequential(
            nn.Conv1d(x_dim+a_dim, args.h_dim, 1), 
            *(
                block for scale in args.scales
                for block in [
                    ResStack(args.h_dim, [
                        nn.Conv1d(args.h_dim, args.h_dim, 5,1,2, groups=args.h_dim)
                        for _ in range(args.block_depth)
                    ]),
                    nn.Conv1d(args.h_dim, args.h_dim, scale, scale),
                ]
            ),
            ResStack(args.h_dim, [
                block for _ in range(args.stack_depth)
                for block in [
                    LinearAttn(args.h_dim, self.rope, args.attn_args),
                    nn.Conv1d(args.h_dim, args.h_dim, 5,1,2, groups=args.h_dim),
                ]
            ]),
            nn.Conv1d(args.h_dim, 1, 1),
        )

    def forward(
        self,
        a: Float[Tensor, "B A L"],
        p: Int[Tensor, "B L"],
        x: Float[Tensor, "B X L"],
    ) -> Float[Tensor, "B"]:
        o = self.net(th.cat([a,x], dim=1))
        return o.squeeze(1).mean(dim=-1)