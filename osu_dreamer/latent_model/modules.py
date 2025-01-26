
from typing import Union
from jaxtyping import Float

from dataclasses import dataclass

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from einops import repeat

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.power_spherical import HypersphericalUniform, PowerSpherical

class ChunkPad(nn.Module):
    def __init__(self, chunk_size: int, pad_value: float = 0.):
        super().__init__()
        self.pad_value = pad_value
        self.chunk_size = chunk_size

    def forward(self, x: Float[Tensor, "B D iL"]) -> Float[Tensor, "B D oL"]:
        pad = (self.chunk_size - x.size(-1)%self.chunk_size) % self.chunk_size
        if pad > 0:
            x = F.pad(x, (0,pad), value=self.pad_value)
        return x
    
def downsample(x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L/2"]:
    D = x.size(1)
    K = th.full((D,1,2), .5).to(x)
    return F.conv1d(x, K, groups=D, stride=2)

def upsample(x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L*2"]:
    D = x.size(1)
    K = th.full((D,1,2), 1.).to(x)
    return F.conv_transpose1d(x, K, groups=D, stride=2)

@dataclass
class EncoderArgs:
    dim: int
    block_depth: int

class Encoder(nn.Module):
    def __init__(self, depth: int, args: EncoderArgs):
        super().__init__()
        self.blocks = nn.ModuleList([
            MP.ResNet([ MP.Seq(args.dim) for _ in range(args.block_depth) ])
            for _ in range(depth+1)
        ])

    def forward(self, x: Float[Tensor, "B D iL"]) -> Float[Tensor, "B D oL"]:
        for i, block in enumerate(self.blocks):
            if i > 0:
                x = downsample(x)
            x = block(x)
        return x

@dataclass
class UNetArgs:
    dim: int
    block_depth: int

class UNet(nn.Module):
    def __init__(
        self,
        depth: int,
        y_dim: int,
        args: UNetArgs,
        in_dim: int = 0,
        out_dim: int = 0,
    ):
        super().__init__()
        self.chunk_size = 1 << depth
        self.y_dim = y_dim
        self.in_dim = in_dim if in_dim != 0 else args.dim
        self.out_dim = out_dim if out_dim != 0 else args.dim

        self.proj_in = nn.Identity() if in_dim == 0 else MP.Conv1d(in_dim, args.dim, 1)
        self.proj_out = nn.Identity() if out_dim == 0 else nn.Conv1d(args.dim, out_dim, 1)

        self.blocks = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Identity() if i==0 else MP.Conv1d(2*args.dim, args.dim, 1),
                    MP.ResNet([ MP.Seq(args.dim) for _ in range(args.block_depth) ]),
                )
                for _ in range(depth+1-i)
            ])
            for i in range(depth+1)
        ])
        self.mix = MP.Conv1d(y_dim+args.dim, args.dim, 1)

    def forward( self,
        x: Float[Tensor, "B {self.in_dim} L"],
        y: Float[Tensor, "B {self.y_dim} zL"],
    ) -> Float[Tensor, "B {self.out_dim} L"]:
        x = self.proj_in(x)

        padding = (self.chunk_size-x.size(-1)%self.chunk_size)%self.chunk_size
        if padding > 0:
            x = F.pad(x, (0, padding))

        xs = []
        for blocks in self.blocks:
            if len(xs) == 0:
                # initialize backbone
                for i, block in enumerate(blocks):
                    if i > 0:
                        x = downsample(x)
                    if i == len(blocks)-1:
                        x = self.mix(MP.cat([x, y], dim=1))
                    x = block(x)
                    xs.append(x)
            else:
                xs = [
                    block(MP.cat([x0, upsample(x1)], dim=1))
                    for block, x0, x1 in zip(blocks, xs, xs[1:])
                ]

        x = xs[0]
        if padding > 0:
            x = x[...,:-padding]
        return self.proj_out(x)
    
class PSVariational(nn.Module):
    def __init__(
        self,
        h_dim: int,
        dim: int,
        net: nn.Module,
    ):
        super().__init__()
        self.net = net
        self.proj_loc = MP.Linear(h_dim, dim)
        self.logscale = nn.Parameter(th.zeros(1))
        self.hs_unif = HypersphericalUniform(dim=dim)

    def forward(
        self,
        x: Float[Tensor, "B iD iL"],
        return_loss: bool = False,
    ) -> Union[
        Float[Tensor, "B oD oL"],
        tuple[
            Float[Tensor, "B oD oL"],
            Float[Tensor, ""],
        ]
    ]:
        x = self.net(x).transpose(1,2) # B L D
        p = PowerSpherical(
            loc = MP.normalize(self.proj_loc(x), dim=-1),
            scale = repeat(self.logscale.exp(), '1 -> b l', b=x.size(0), l=x.size(1)),
        )

        z = p.rsample().transpose(1,2)
        if not return_loss:
            return z
        
        kl_div = th.distributions.kl_divergence(p, self.hs_unif)
        return z, kl_div.mean()