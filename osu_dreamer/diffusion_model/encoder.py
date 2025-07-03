
from dataclasses import dataclass

from torch import nn

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.spec_features import SpecFeatures

@dataclass
class EncoderArgs:
    h_dim: int

class Encoder(nn.Sequential):
    def __init__(
        self,
        dim: int,
        args: EncoderArgs,
    ):
        super().__init__(
            SpecFeatures(args.h_dim),
            MP.Conv1d(args.h_dim, dim, 1),
        )