
from jaxtyping import Float

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
    def __init__(self, dim: int, depth: int, blocks_per_depth: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            MP.ResNet([ MP.Seq(dim) for _ in range(blocks_per_depth) ])
            for _ in range(depth)
        ])

    def forward(self, x: Float[Tensor, "B D iL"]) -> Float[Tensor, "B D oL"]:
        D = x.size(1)
        f = th.tensor([.5,.5], device=x.device)[None,None].repeat(D,1,1)
        for block in self.blocks:
            x = F.conv1d(x, f, groups=D, stride=2)
            x = block(x)
        return x

class Decoder(nn.Module):
    def __init__(self, dim: int, depth: int, blocks_per_depth: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            MP.ResNet([ MP.Seq(dim) for _ in range(blocks_per_depth) ])
            for _ in range(depth)
        ])

    def forward(self, h: Float[Tensor, "B D iL"]) -> Float[Tensor, "B D oL"]:
        D = h.size(1)
        f = 2 * th.tensor([.5,.5], device=h.device)[None,None].repeat(D,1,1)
        for block in self.blocks:
            h = F.conv_transpose1d(h, f, groups=D, stride=2)
            h = block(h)
        return h