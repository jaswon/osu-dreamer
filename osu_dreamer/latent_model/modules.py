
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

@dataclass
class EncoderArgs:
    dim: int
    blocks_per_depth: int

class Encoder(nn.Module):
    def __init__(self, depth: int, args: EncoderArgs, *, down: bool):
        super().__init__()
        resample = nn.Conv1d if down else nn.ConvTranspose1d
        self.resamples = nn.ModuleList([
            resample(args.dim, args.dim, 4,2,1, groups=args.dim)
            for _ in range(depth)
        ])
        self.down = down
        self.blocks = nn.ModuleList([
            MP.ResNet([ MP.Seq(args.dim) for _ in range(args.blocks_per_depth) ])
            for _ in range(depth)
        ])

    def forward(self, x: Float[Tensor, "B D iL"]) -> Float[Tensor, "B D oL"]:
        for resample, block in zip(self.resamples, self.blocks):
            x = resample(x)
            x = block(x)
        return x
    
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