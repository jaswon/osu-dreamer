
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
from torch import nn, Tensor 

import osu_dreamer.modules.mp as MP

@dataclass
class AudioFeatureArgs:
    dim: int
    depth: int
    expand: int

class AudioFeatures(nn.Module):
    def __init__(
        self,
        dim: int,
        args: AudioFeatureArgs,
    ):
        super().__init__()

        self.proj_in = MP.Conv1d(dim+1, args.dim, 1)
        self.net = MP.ResNet([ MP.Seq(args.dim, args.expand) for _ in range(args.depth) ])

    def forward(
        self,
        audio: Float[Tensor, "B A L"],
    ) -> Float[Tensor, "B D L"]:
        audio = MP.cat([audio, th.ones_like(audio[:,:1,:])], dim=1)
        h = self.proj_in(audio)
        return self.net(h)