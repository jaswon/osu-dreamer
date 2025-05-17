
from jaxtyping import Float

from dataclasses import dataclass

import torch as th
from torch import nn, Tensor

from osu_dreamer.data.labels import NUM_LABELS

import osu_dreamer.modules.mp as MP

@dataclass
class LabelEmbeddingArgs:
    features: int

class LabelEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        args: LabelEmbeddingArgs,
    ):
        super().__init__()
        self.null = nn.Parameter(th.randn(NUM_LABELS, args.features))

        # random fourier features
        self.register_buffer('f', 2 * th.pi * th.randn(NUM_LABELS, args.features))
        self.register_buffer('p', 2 * th.pi * th.rand(NUM_LABELS, args.features))

        self.out = nn.Sequential(
            MP.Linear(NUM_LABELS * args.features, dim),
            MP.SiLU(),
        )

    def forward(self, x: Float[Tensor, str(f"B {NUM_LABELS}")]) -> Float[Tensor, "B E"]:
        return self.out(th.where(
            x[:,:,None] < 0,
            self.null[None],
            2**.5 * th.cos(x[:,:,None] * self.f + self.p),
        ).view(x.size(0), -1).float())