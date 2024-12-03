
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


def resample(x: Float[Tensor, "B D iL"], up: bool) -> Float[Tensor, "B D oL"]:
    f = th.tensor([1,1], dtype=th.float, device=x.device)
    f = f / f.sum()
    c = x.size(1)
    f = f[None,None].repeat(c,1,1)
    if up:
        return F.conv_transpose1d(x, f*2, groups=c, stride=2)
    return F.conv1d(x, f, groups=c, stride=2)

class Encoder(nn.Module):
    def __init__(self, dim: int, depth: int, blocks_per_depth: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(*( Block(dim) for _ in range(blocks_per_depth) ))
            for _ in range(depth)
        ])

    def forward(self, x: Float[Tensor, "B D iL"]) -> Float[Tensor, "B D oL"]:
        for block in self.blocks:
            x = resample(x, up=False)
            x = block(x)
        return x

class Decoder(nn.Module):
    def __init__(self, dim: int, depth: int, blocks_per_depth: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(*( Block(dim) for _ in range(blocks_per_depth) ))
            for _ in range(depth)
        ])

    def forward(self, h: Float[Tensor, "B D iL"]) -> Float[Tensor, "B D oL"]:
        for block in self.blocks:
            h = resample(h, up=True)
            h = block(h)
        return h

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        expand: int = 1,
    ):
        H = dim * expand
        super().__init__()
        self.hg = nn.Sequential(
            MP.Conv1d(dim, H*2, 1),
            MP.SiLU(),
        )
        self.net = nn.Sequential(
            MP.Conv1d(H, H, 3,1,1, groups=H),
            MP.SiLU(),
            MP.minGRU2(H),
        )
        self.out = MP.Conv1d(H, dim, 1)

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L"]:
        x = MP.pixel_norm(x)
        h,g = self.hg(x).chunk(2, dim=1)
        o = self.out(self.net(h) * g)
        return MP.add(x, o, t=.3)