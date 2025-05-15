
from dataclasses import dataclass
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.mingru import MinGRU


class sequenceMixer(nn.Module):
    def __init__(self, dim: int, expand: int):
        super().__init__()
        h_dim = dim // 2
        assert h_dim * 2 == dim
        self.fore = MinGRU(dim, out_dim = h_dim * expand)
        self.back = MinGRU(dim, out_dim = h_dim * expand)
        self.out = MP.Conv1d(dim * expand, dim, 1)

    def forward(
        self,
        x: Float[Tensor, "B X L"],
    ) -> Float[Tensor, "B X L"]:
        return self.out(MP.cat([
            self.fore(x), 
            self.back(x.flip(2)).flip(2),
        ], dim=1))
    
class channelMixer(nn.Module):
    def __init__(self, dim: int, expand: int):
        super().__init__()
        self.proj_in = MP.Conv1d(dim, dim, 1)
        self.proj_h = nn.Sequential(
            MP.Conv1d(dim, dim, 3,1,1, groups=dim),
            MP.Conv1d(dim, dim * expand, 1),
        )
        self.proj_g = nn.Sequential(
            MP.Conv1d(dim, dim, 3,1,1, groups=dim),
            MP.Conv1d(dim, dim * expand, 1),
            MP.SiLU(),
        )
        self.proj_out = MP.Conv1d(dim * expand, dim, 1)

    def forward(
        self,
        x: Float[Tensor, "B X L"],
    ) -> Float[Tensor, "B X L"]:
        x = self.proj_in(x)
        h,g = self.proj_h(x), self.proj_g(x)
        return self.proj_out(h*g)
    
class condBlock(nn.Module):
    def __init__(self, dim: int, c_dim: int, op: nn.Module):
        super().__init__()
        self.op = op
        self.scale = nn.Sequential(
            MP.Linear(c_dim, dim),
            MP.Gain(),
        )
        self.shift = nn.Sequential(
            MP.Linear(c_dim, dim),
            MP.Gain(),
        )
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
        r = r * (1+self.scale(c)[:,:,None]) + self.shift(c)[:,:,None]
        r = self.op(r)
        r = self.alpha(c)[:,:,None] * r
        return x + r
    
class uncondBlock(nn.Module):
    def __init__(self, dim: int, _: None, op: nn.Module):
        super().__init__()
        self.op = op
        self.scale = nn.Parameter(th.zeros(dim, 1))
        self.shift = nn.Parameter(th.zeros(dim, 1))
        self.alpha = nn.Parameter(th.zeros(dim, 1))

    def forward(
        self, 
        x: Float[Tensor, "B X L"],
        _: None,
    ) -> Float[Tensor, "B X L"]:
        r = MP.pixel_norm(x)
        r = r * (1+self.scale) + self.shift
        r = self.op(r)
        r = self.alpha * r
        return x + r
    
class DiTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        c_dim: None | int,
        expand: int,
    ):
        super().__init__()
        if c_dim is None:
            block = lambda op: uncondBlock(dim, None, op)
        else:
            block = lambda op: condBlock(dim, c_dim, op)
        self.seq_mixer = block(sequenceMixer(dim, expand))
        self.chn_mixer = block(channelMixer(dim, expand))

    def forward(
        self,
        x: Float[Tensor, "B X L"],
        c: None | Float[Tensor, "B C"],
    ) -> Float[Tensor, "B X L"]:
        x = self.seq_mixer(x,c)
        x = self.chn_mixer(x,c)
        return x
    

@dataclass
class DiTArgs:
    depth: int
    expand: int

class DiT(nn.Module):
    def __init__(
        self,
        dim: int,
        c_dim: None | int,
        args: DiTArgs,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            DiTBlock(dim, c_dim, args.expand)
            for _ in range(args.depth)
        ])

    def forward(
        self,
        x: Float[Tensor, "B X L"],
        c: None | Float[Tensor, "B C"] = None,
    ) -> Float[Tensor, "B X L"]:
        for block in self.blocks:
            x = checkpoint(block, x, c, use_reentrant=False) # type: ignore
        return x