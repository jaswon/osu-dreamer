
from typing import Union
from jaxtyping import Float

from dataclasses import dataclass

from functools import partial

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

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
        self.down = down
        self.blocks = nn.ModuleList([
            MP.ResNet([ MP.Seq(args.dim) for _ in range(args.blocks_per_depth) ])
            for _ in range(depth)
        ])

    def forward(self, x: Float[Tensor, "B D iL"]) -> Float[Tensor, "B D oL"]:
        D = x.size(1)
        f = th.tensor([.5,.5], device=x.device)[None,None].repeat(D,1,1)
        resample = (
            partial(F.conv1d, weight=f)
            if self.down else
            partial(F.conv_transpose1d, weight=2*f)
        )

        for block in self.blocks:
            x = resample(x, groups=D, stride=2)
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
        self.proj_logscale = nn.Sequential(
            MP.Linear(h_dim, 1),
            MP.Gain(),
        )
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
            scale = self.proj_logscale(x).squeeze(-1).exp(),
        )

        z = p.rsample().transpose(1,2)
        if not return_loss:
            return z
        
        kl_div = th.distributions.kl_divergence(p, self.hs_unif)
        return z, kl_div.mean()