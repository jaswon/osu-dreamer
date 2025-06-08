
from jaxtyping import Float

from dataclasses import dataclass

from torch import nn, Tensor

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.dit import DiT, DiTArgs

@dataclass
class EncoderArgs:
    depth: int
    expand: int

class Encoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        dim: int,
        args: EncoderArgs,
    ):
        super().__init__()
        self.proj_in = MP.Conv1d(in_dim, dim, 1)
        self.net = DiT(dim, None, DiTArgs(args.depth, args.expand))

    def forward(self, x: Float[Tensor, "B A L"],) -> Float[Tensor, "B D L"]:
        return self.net(self.proj_in(x))