
from jaxtyping import Float

from dataclasses import dataclass

import torch as th
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

import osu_dreamer.modules.mp as MP

from .deltanet import DeltaNet, DeltaNetArgs
    
class channelMixer(nn.Module):
    def __init__(self, dim: int, expand: int):
        super().__init__()
        self.proj_h = MP.Linear(dim, dim*expand)
        self.proj_g = MP.Linear(dim, dim*expand)
        self.proj_out = MP.Linear(dim * expand, dim)

    def forward(self, x: Float[Tensor, "B N E"]) -> Float[Tensor, "B N E"]:
        h,g = self.proj_h(x), MP.silu(self.proj_g(x))
        return self.proj_out(h*g)
    
class condBlock(nn.Module):
    def __init__(self, dim: int, op: nn.Module):
        super().__init__()
        self.op = op
        self.alpha = nn.Parameter(th.zeros(dim))

    def forward(self, x: Float[Tensor, "B N E"], *args) -> Float[Tensor, "B N E"]:
        r = MP.normalize(x, dim=-1)
        r = self.op(r, *args)
        r = self.alpha * r
        return x + r
    
class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        expand: int,
        deltanet_args: DeltaNetArgs,
    ):
        super().__init__()
        self.seq_mixer = condBlock(dim, DeltaNet(dim, deltanet_args))
        self.chn_mixer = condBlock(dim, channelMixer(dim, expand))

    def forward(self, x: Float[Tensor, "B N E"]) -> Float[Tensor, "B N E"]:
        x = self.seq_mixer(x)
        x = self.chn_mixer(x)
        return x

@dataclass
class EncoderArgs:
    depth: int
    expand: int
    deltanet_args: DeltaNetArgs

class Encoder(nn.Module):
    def __init__(
        self,
        dim: int,
        args: EncoderArgs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            EncoderBlock(dim, args.expand, args.deltanet_args)
            for _ in range(args.depth)
        ])

    def forward(self, x: Float[Tensor, "B N E"]) -> Float[Tensor, "B N E"]:
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) # type: ignore
        return x