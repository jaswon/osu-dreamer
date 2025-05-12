
from jaxtyping import Float

from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.mingru import MinGRU


class sequenceMixer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        h_dim = dim // 2
        assert h_dim * 2 == dim
        self.fore = MinGRU(dim, out_dim = h_dim)
        self.back = MinGRU(dim, out_dim = h_dim)
        self.out = MP.Conv1d(dim, dim, 1)

    def forward(
        self,
        x: Float[Tensor, "B X L"],
    ) -> Float[Tensor, "B X L"]:
        return self.out(MP.cat([self.fore(x), self.back(x)], dim=1))
    
class channelMixer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj_in = MP.Conv1d(dim, dim, 1)
        self.proj_h = nn.Sequential(
            MP.Conv1d(dim, dim, 3,1,1, groups=dim),
            MP.Conv1d(dim, dim, 1),
        )
        self.proj_g = nn.Sequential(
            MP.Conv1d(dim, dim, 3,1,1, groups=dim),
            MP.Conv1d(dim, dim, 1),
            MP.SiLU(),
        )
        self.proj_out = MP.Conv1d(dim, dim, 1)

    def forward(
        self,
        x: Float[Tensor, "B X L"],
    ) -> Float[Tensor, "B X L"]:
        x = self.proj_in(x)
        h,g = self.proj_h(x), self.proj_g(x)
        return self.proj_out(h*g)
    
class block(nn.Module):
    def __init__(self, dim: int, c_dim: int, op: nn.Module):
        super().__init__()
        self.op = op
        self.scale = MP.Linear(c_dim, dim)
        self.shift = MP.Linear(c_dim, dim)
        self.alpha = nn.Sequential(
            MP.Linear(c_dim, dim),
            MP.Gain(),
        )

    def forward(
        self,
        x: Float[Tensor, "B X L"],
        c: Float[Tensor, "B C"],
    ) -> Float[Tensor, "B X L"]:
        r = MP.pixel_norm(x)
        r = r * self.scale(c)[:,:,None] + self.shift(c)[:,:,None]
        r = self.op(r)
        r = self.alpha(c)[:,:,None] * r
        return r
    
class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        c_dim: int,
    ):
        super().__init__()
        self.seq_mixer = block(dim, c_dim, sequenceMixer(dim))
        self.chn_mixer = block(dim, c_dim, channelMixer(dim))

    def forward(
        self,
        x: Float[Tensor, "B X L"],
        c: Float[Tensor, "B C"],
    ) -> Float[Tensor, "B X L"]:
        x = x + self.seq_mixer(x,c)
        x = x + self.chn_mixer(x,c)
        return x
    
class DiT(nn.Module):
    def __init__(
        self,
        dim: int,
        c_dim: int,
        depth: int,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([ DiTBlock(dim, c_dim) for _ in range(depth) ])

    def forward(
        self,
        x: Float[Tensor, "B X L"],
        c: Float[Tensor, "B C"],
    ) -> Float[Tensor, "B X L"]:
        for block in self.blocks:
            x = checkpoint(block, x, c, use_reentrant=False)
        return x