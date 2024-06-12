
from dataclasses import dataclass

from osu_dreamer.common.residual import ResStack
from osu_dreamer.common.s4d import S4Block, S4Args
from osu_dreamer.common.unet import UNet

@dataclass
class EncoderArgs:
    scales: list[int]
    block_depth: int
    stack_depth: int
    ssm_args: S4Args

class Encoder(UNet):
    def __init__(self, dim: int, args: EncoderArgs):
        super().__init__(
            dim, 
            args.scales, 
            args.block_depth,
            ResStack(dim, [
                S4Block(dim, args.ssm_args)
                for _ in range(args.stack_depth)
            ]),
        )