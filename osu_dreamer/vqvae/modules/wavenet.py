
from jaxtyping import Float

from dataclasses import dataclass

from torch import nn, Tensor

import osu_dreamer.modules.mp as MP

from .se import SqueezeExcitation

X = Float[Tensor, "B X L"]

class WaveNetLayer(nn.Module):
    def __init__(self, dim: int, squeeze: int, d: int):
        super().__init__()
        self.filter = MP.Conv1d(dim, dim, 3, dilation=2**d, padding=2**d, groups=dim)
        self.gate = MP.Conv1d(dim, dim, 3, dilation=2**d, padding=2**d, groups=dim)
        self.res = MP.Conv1d(dim, dim, 1)
        self.se = SqueezeExcitation(dim, squeeze)

    def forward(self, x: X) -> X:
        res = self.res(self.filter(x).tanh() / .627 * MP.sigmoid(self.gate(x)))
        return MP.add(x, self.se(res))
    
@dataclass
class WaveNetArgs:
    num_stacks: int
    stack_depth: int
    squeeze: int

class WaveNet(nn.Sequential):
    def __init__(
        self,
        dim: int,
        args: WaveNetArgs,
    ):
        super().__init__(*[
            WaveNetLayer(dim, args.squeeze, d)
            for _ in range(args.num_stacks)
            for d in range(args.stack_depth)
        ])