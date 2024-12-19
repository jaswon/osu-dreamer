
from jaxtyping import Float

from functools import partial

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from vector_quantize_pytorch import VectorQuantize

import osu_dreamer.modules.mp as MP

class VectorQuantizer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        decay: float = 0.8,
        commitment: float = 0.25,
    ):
        super().__init__()
        self.vq = VectorQuantize(
            dim = embedding_dim,
            codebook_size = num_embeddings,
            decay = decay,
            commitment_weight = commitment,
            rotation_trick = True,
            use_cosine_sim = True,
        )

    def forward(
        self,
        x: Float[Tensor, "B D L"],
    ) -> tuple[
        Float[Tensor, "B D L"],
        Float[Tensor, ""],
    ]:
        q, _, commit_loss = self.vq(x.transpose(1,2))
        return q.transpose(1,2), commit_loss[0]

class Encoder(nn.Module):
    def __init__(self, dim: int, depth: int, blocks_per_depth: int, down: bool):
        super().__init__()
        self.down = down
        self.blocks = nn.ModuleList([
            MP.ResNet([ MP.Seq(dim) for _ in range(blocks_per_depth) ])
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