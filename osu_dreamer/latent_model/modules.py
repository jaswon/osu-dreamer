
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from einops import rearrange

import osu_dreamer.modules.mp as MP

class VectorQuantizer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: int,
        commitment: float = 1.
    ):
        super().__init__()
        self.commitment = commitment
        self.embeddings = nn.Parameter(th.randn(num_embeddings, embedding_dim))

    def forward(
        self,
        x: Float[Tensor, "B D L"],
    ) -> tuple[
        Float[Tensor, "B D L"],
        Float[Tensor, ""],
    ]:
        b = x.size(0)
        x = rearrange(x, 'b d l -> (b l) d')
        x = F.normalize(x, p=2, dim=1)
        cb = MP.get_normed_weight(self.embeddings, self.training)
        quantized = cb[th.argmax(x @ cb.t(), dim=1)]

        e_loss = 1 - (quantized.detach() * x).sum(dim=1)
        q_loss = 1 - (quantized * x.detach()).sum(dim=1)
        loss = self.commitment * e_loss + q_loss

        quantized = x + (quantized - x).detach()
        quantized = rearrange(quantized, "(b l) d -> b d l", b=b)

        return quantized, loss.mean()

class Encoder(nn.Module):
    def __init__(self, dim: int, depth: int, blocks_per_depth: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            MP.ResNet([ MP.Seq(dim, dim) for _ in range(blocks_per_depth) ])
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
            MP.ResNet([ MP.Seq(dim, dim) for _ in range(blocks_per_depth) ])
            for _ in range(depth)
        ])

    def forward(self, h: Float[Tensor, "B D iL"]) -> Float[Tensor, "B D oL"]:
        D = h.size(1)
        f = 2 * th.tensor([.5,.5], device=h.device)[None,None].repeat(D,1,1)
        for block in self.blocks:
            h = F.conv_transpose1d(h, f, groups=D, stride=2)
            h = block(h)
        return h