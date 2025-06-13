
from jaxtyping import Float

from dataclasses import dataclass

from torch import nn, Tensor

import osu_dreamer.modules.mp as MP

from .wavenet import WaveNet, WaveNetArgs

@dataclass
class AutoEncoderArgs:
    h_dim: int
    depth: int
    stride: int
    layer_args: WaveNetArgs

class Encoder(nn.Module):
    def __init__(
        self,
        dim: int,
        args: AutoEncoderArgs,
    ):
        super().__init__()
        self.proj_in = MP.Conv1d(dim, args.h_dim, 1) if dim != args.h_dim else nn.Identity()
        self.proj_out = MP.Conv1d(args.h_dim, dim, 1) if dim != args.h_dim else nn.Identity()
        self.blocks = nn.ModuleList([
            WaveNet(args.h_dim, args.layer_args)
            for _ in range(args.depth)
        ])
        self.downs = nn.ModuleList([
            nn.Sequential(
                MP.Conv1d(
                    args.h_dim, args.h_dim, 
                    args.stride, 1, 'same', 
                    groups=args.h_dim,
                ),
                nn.AvgPool1d(args.stride),
            )
            for _ in range(args.depth)
        ])

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D l"]:
        x = self.proj_in(x)
        for block, down in zip(self.blocks, self.downs):
            x = block(x)
            x = down(x)
        return self.proj_out(x)
    

class Decoder(nn.Module):
    def __init__(
        self,
        dim: int,
        args: AutoEncoderArgs,
    ):
        super().__init__()
        self.proj_in = MP.Conv1d(dim, args.h_dim, 1) if dim != args.h_dim else nn.Identity()
        self.proj_out = MP.Conv1d(args.h_dim, dim, 1) if dim != args.h_dim else nn.Identity()
        self.blocks = nn.ModuleList([
            WaveNet(args.h_dim, args.layer_args)
            for _ in range(args.depth)
        ])
        self.ups = nn.ModuleList([
            nn.Sequential(
                nn.Upsample(scale_factor=args.stride),
                MP.Conv1d(
                    args.h_dim, args.h_dim, 
                    args.stride, 1, 'same', 
                    groups=args.h_dim,
                ),
            )
            for _ in range(args.depth)
        ])

    def forward(self, x: Float[Tensor, "B D l"]) -> Float[Tensor, "B D L"]:
        x = self.proj_in(x)
        for block, up in zip(self.blocks, self.ups):
            x = up(x)
            x = block(x)
        return self.proj_out(x)
    
