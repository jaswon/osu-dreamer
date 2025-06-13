
from jaxtyping import Float

from dataclasses import dataclass

import torch as th
from torch import nn, Tensor

import osu_dreamer.modules.mp as MP

X = Float[Tensor, "B X L"]

class WaveNetLayer(nn.Module):
    def __init__(self, dim: int, d: int):
        super().__init__()
        self.filter = MP.Conv1d(dim, dim, 3, dilation=2**d, padding=2**d, groups=dim)
        self.gate = MP.Conv1d(dim, dim, 3, dilation=2**d, padding=2**d, groups=dim)
        self.res = MP.Conv1d(dim, dim, 1)
        self.alpha = nn.Parameter(th.zeros(dim, 1))

    def forward(self, x: X) -> X:
        res = self.res(self.filter(x).tanh() / .627 * self.gate(x).sigmoid() / .542)
        return MP.add(x, self.alpha * res)
    
@dataclass
class WaveNetArgs:
    num_stacks: int
    stack_depth: int

class WaveNet(nn.Sequential):
    def __init__(
        self,
        dim: int,
        args: WaveNetArgs,
    ):
        super().__init__(*[
            WaveNetLayer(dim, d)
            for _ in range(args.num_stacks)
            for d in range(args.stack_depth)
        ])