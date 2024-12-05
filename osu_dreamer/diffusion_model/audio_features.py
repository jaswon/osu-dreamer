
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
from torch import nn, Tensor 

from osu_dreamer.data.load_audio import A_DIM

import osu_dreamer.modules.mp as MP
from .modules import Seq, ResNet

@dataclass
class AudioFeatureArgs:
    dim: int
    depth: int
    expand: int

class AudioFeatures(nn.Module):
    def __init__(
        self,
        args: AudioFeatureArgs,
    ):
        super().__init__()

        self.proj_in = MP.Conv1d(A_DIM+1, args.dim, 1)
        self.net = ResNet([ Seq(args.dim, args.dim * args.expand) for _ in range(args.depth) ])

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
    ) -> Float[Tensor, "B D L"]:
        audio = MP.cat([audio, th.ones_like(audio[:,:1,:])], dim=1)
        h = self.proj_in(audio)
        return self.net(h)