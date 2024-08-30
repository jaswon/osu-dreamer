
from dataclasses import dataclass

from torch import nn

from .residual import ResStack

@dataclass
class WaveNetArgs:
    num_blocks: int
    block_depth: int

class WaveNet(ResStack):
    """context length = num_blocks*2^block_depth)"""
    def __init__(self, dim: int, args: WaveNetArgs):
        super().__init__(dim, [
            nn.Sequential(
                nn.ZeroPad1d((1,0) if d==0 else 2**(d-1)),
                nn.Conv1d(dim, dim, 2, dilation=2**d, groups=dim),
                nn.SiLU(),
            )
            for _ in range(args.num_blocks)
            for d in range(args.block_depth)
        ]) 