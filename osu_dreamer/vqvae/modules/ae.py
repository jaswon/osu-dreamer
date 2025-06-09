
from jaxtyping import Float

from torch import nn, Tensor

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.dit import DiTBlock

class Encoder(nn.Module):
    def __init__(
        self,
        dim: int,
        h_dim: int,
        depth: int,
        stride: int,
    ):
        super().__init__()
        self.proj_in = MP.Conv1d(dim, h_dim, 1)
        self.proj_out = MP.Conv1d(h_dim, dim, 1)
        self.blocks = nn.ModuleList([
            DiTBlock(h_dim, None, 1)
            for _ in range(depth)
        ])
        self.downs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    h_dim, h_dim, 
                    stride+2, stride, 1,
                    groups=h_dim, bias=False,
                ),
                MP.Conv1d(h_dim, h_dim, 1),
                MP.SiLU(),
            )
            for _ in range(depth)
        ])

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D l"]:
        x = self.proj_in(x)
        for block, down in zip(self.blocks, self.downs):
            x = block(x, None)
            x = down(x)
        return self.proj_out(x)
    

class Decoder(nn.Module):
    def __init__(
        self,
        dim: int,
        h_dim: int,
        depth: int,
        stride: int,
    ):
        super().__init__()
        self.proj_in = MP.Conv1d(dim, h_dim, 1)
        self.proj_out = MP.Conv1d(h_dim, dim, 1)
        self.blocks = nn.ModuleList([
            DiTBlock(h_dim, None, 1)
            for _ in range(depth)
        ])
        self.ups = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(
                    h_dim, h_dim, 
                    stride+2, stride, 1,
                    groups=h_dim, bias=False,
                ),
                MP.Conv1d(h_dim, h_dim, 1),
                MP.SiLU(),
            )
            for _ in range(depth)
        ])

    def forward(self, x: Float[Tensor, "B D l"]) -> Float[Tensor, "B D L"]:
        x = self.proj_in(x)
        for block, up in zip(self.blocks, self.ups):
            x = block(x, None)
            x = up(x)
        return self.proj_out(x)