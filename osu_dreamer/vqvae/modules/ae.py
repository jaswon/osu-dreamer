
from jaxtyping import Float

from dataclasses import dataclass

from torch import nn, Tensor

import osu_dreamer.modules.mp as MP

from .wavenet import WaveNet, WaveNetArgs

@dataclass
class AutoEncoderArgs:
    depth: int
    stride: int
    expand: int
    layer_args: WaveNetArgs

class Encoder(nn.Module):
    def __init__(
        self,
        dim: int,
        args: AutoEncoderArgs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            WaveNet(dim, args.layer_args)
            for _ in range(args.depth)
        ])
        h_dim = dim * args.expand
        self.downs = nn.ModuleList([
            nn.Sequential(
                MP.Conv1d(dim, h_dim, 1),
                MP.Conv1d(h_dim, h_dim, args.stride*2, 1, 'same', groups=h_dim),
                nn.AvgPool1d(args.stride),
                MP.SiLU(),
                MP.Conv1d(h_dim, dim, 1),
            )
            for _ in range(args.depth)
        ])

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D l"]:
        for block, down in zip(self.blocks, self.downs):
            x = block(x)
            x = down(x)
        return x
    

class Decoder(nn.Module):
    def __init__(
        self,
        dim: int,
        args: AutoEncoderArgs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            WaveNet(dim, args.layer_args)
            for _ in range(args.depth)
        ])
        h_dim = dim * args.expand
        self.ups = nn.ModuleList([
            nn.Sequential(
                MP.Conv1d(dim, h_dim, 1),
                nn.Upsample(scale_factor=args.stride, mode='linear'),
                MP.Conv1d(h_dim, h_dim, args.stride*2, 1, 'same', groups=h_dim),
                MP.SiLU(),
                MP.Conv1d(h_dim, dim, 1),
            )
            for _ in range(args.depth)
        ])

    def forward(self, x: Float[Tensor, "B D l"]) -> Float[Tensor, "B D L"]:
        for block, up in zip(self.blocks, self.ups):
            x = up(x)
            x = block(x)
        return x
    
