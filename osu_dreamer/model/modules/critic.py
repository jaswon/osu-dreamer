
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from osu_dreamer.common.residual import ResStack

from osu_dreamer.data.beatmap.encode import CURSOR_DIM, HIT_DIM

CURSOR_FEATURES = 4
def cursor_features(cursor: Float[Tensor, str(f"B {CURSOR_DIM} L")]) -> Float[Tensor, str(f"B {CURSOR_FEATURES} L")]:
    cursor_diff = F.pad(cursor[...,1:] - cursor[...,:-1], (1,0), mode='replicate')
    return th.cat([ cursor, cursor_diff ], dim=1)

@dataclass
class CriticArgs:
    h_dim: int
    stack_depth: int
    wave_depth: int

class Critic(nn.Module):
    def __init__(
        self,
        args: CriticArgs,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(HIT_DIM + CURSOR_FEATURES, args.h_dim, 1),
            ResStack(args.h_dim, [
                nn.Sequential(
                    nn.ZeroPad1d((2**d,0)),
                    nn.Conv1d(args.h_dim, 2*args.h_dim, 2, dilation=2**d),
                    nn.GLU(dim=1),
                )
                for i in range(args.stack_depth)
                for d in [i % args.wave_depth]
            ]), # wave net
            nn.Conv1d(args.h_dim, CURSOR_DIM, 1),
        )

    def forward(
        self, 
        hit: Float[Tensor, str(f"B {HIT_DIM} L")],
        cursor: Float[Tensor, str(f"B {CURSOR_DIM} L")],
    ) -> Float[Tensor, str(f"B {CURSOR_DIM} L")]:
        return self.net(th.cat([hit, cursor_features(cursor)], dim=1))