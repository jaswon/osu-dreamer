
from dataclasses import dataclass

from torch import nn

from osu_dreamer.common.residual import ResStack
from osu_dreamer.common.s4d import S4Block, S4Args
from osu_dreamer.common.unet import UNet

@dataclass
class EncoderArgs:
    h_dim: int
    unet_scales: list[int]
    stack_depth: int
    ssm_args: S4Args

class Encoder(nn.Sequential):
    def __init__(self, a_dim: int, args: EncoderArgs):
        super().__init__(
            nn.Conv1d(a_dim, args.h_dim, 1),
            UNet(
                args.h_dim, 
                args.unet_scales, 
                ResStack(args.h_dim, [
                    S4Block(args.h_dim, args.ssm_args)
                    for _ in range(args.stack_depth)
                ]),
            ),
        )