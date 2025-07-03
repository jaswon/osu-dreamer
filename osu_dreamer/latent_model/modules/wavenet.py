
from jaxtyping import Float

from dataclasses import dataclass

from torch import nn, Tensor

import osu_dreamer.modules.mp as MP

X = Float[Tensor, "B X L"]

class WaveNetLayer(nn.Module):
    def __init__(self, dim: int, d: int, expand: int):
        super().__init__()
        h_dim = dim * expand
        self.filter = nn.Sequential(
            MP.Conv1d(dim, h_dim, 1),
            MP.Conv1d(h_dim, h_dim, 3, dilation=d, padding=d, groups=h_dim),
        )
        self.gate = nn.Sequential(
            MP.Conv1d(dim, h_dim, 1),
            MP.Conv1d(h_dim, h_dim, 3, dilation=d, padding=d, groups=h_dim),
        )
        self.res = MP.Conv1d(h_dim, dim, 1)
        self.skip = MP.Conv1d(h_dim, dim, 1)

    def forward(self, x: X) -> tuple[X,X]:
        h = self.filter(x) * MP.sigmoid(self.gate(x))
        return self.res(h), self.skip(h)
    
@dataclass
class WaveNetArgs:
    num_stacks: int
    stack_depth: int
    expand: int = 1

class WaveNet(nn.Module):
    def __init__(
        self,
        dim: int,
        args: WaveNetArgs,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            WaveNetLayer(dim, 2**d, args.expand)
            for _ in range(args.num_stacks)
            for d in range(args.stack_depth)
        ])
        self.out = nn.Sequential(
            MP.Sum(len(self.layers)),
            MP.SiLU(),
            MP.Conv1d(dim, dim, 1),
        )

    def forward(self, x: X) -> X:
        skips = []
        for layer in self.layers:
            res, skip = layer(x)
            x = MP.add(x,res,t=.3)
            skips.append(skip)
        return self.out(skips)