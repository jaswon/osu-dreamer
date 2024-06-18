
from dataclasses import dataclass

from torch import nn

from osu_dreamer.common.residual import ResStack
from osu_dreamer.common.s4d import S4Block, S4Args
from osu_dreamer.common.unet import UNet

@dataclass
class EncoderArgs:
    scales: list[int]
    block_depth: int
    stack_depth: int
    ssm_args: S4Args

class Encoder(ResStack):
    def __init__(self, dim: int, args: EncoderArgs):
        super().__init__(dim, [
            UNet(
                dim, 
                args.scales, 
                lambda: ResStack(dim, [
                    nn.Sequential(
                        nn.Conv1d(dim, dim, 3,1,1, groups=dim),
                        nn.GroupNorm(1, dim),
                        nn.SiLU(),
                    )
                    for _ in range(args.block_depth)
                ]),
                S4Block(dim, args.ssm_args),
            )
            for _ in range(args.stack_depth)
        ])

