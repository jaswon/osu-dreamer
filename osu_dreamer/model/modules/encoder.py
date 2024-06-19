
from dataclasses import dataclass

from torch import nn

from osu_dreamer.common.residual import ResStack
from osu_dreamer.common.s4d import S4Block, S4Args

@dataclass
class EncoderArgs:
    stack_depth: int
    ssm_args: S4Args

class Encoder(nn.Sequential):
    def __init__(self, in_dim: int, dim: int, args: EncoderArgs):
        super().__init__(
            nn.Conv1d(in_dim, dim, 1),
            ResStack(dim, [
                S4Block(dim, args.ssm_args)
                for _ in range(args.stack_depth)
            ]),
        )

