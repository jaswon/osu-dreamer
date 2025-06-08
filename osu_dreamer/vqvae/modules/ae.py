
from jaxtyping import Float

from torch import nn, Tensor

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.dit import DiTBlock

class Encoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        stride: int,
        expand: int,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            DiTBlock(dim, None, expand)
            for _ in range(depth)
        ])
        self.downs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    dim, dim, 
                    stride+2, stride, 1,
                    groups=dim, bias=False,
                ),
                MP.Conv1d(dim, dim, 1),
                MP.SiLU(),
            )
            for _ in range(depth)
        ])

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D l"]:
        for block, down in zip(self.blocks, self.downs):
            x = block(x, None)
            x = down(x)
        return x
    

class Decoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        stride: int,
        expand: int,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            DiTBlock(dim, None, expand)
            for _ in range(depth)
        ])
        self.ups = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(
                    dim, dim, 
                    stride+2, stride, 1,
                    groups=dim, bias=False,
                ),
                MP.Conv1d(dim, dim, 1),
                MP.SiLU(),
            )
            for _ in range(depth)
        ])

    def forward(self, x: Float[Tensor, "B D l"]) -> Float[Tensor, "B D L"]:
        for block, up in zip(self.blocks, self.ups):
            x = block(x, None)
            x = up(x)
        return x