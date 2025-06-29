
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from einops import rearrange
from einops.layers.torch import Rearrange

import osu_dreamer.modules.mp as MP

class fConv(nn.Module):
    def __init__(self, h_dim, *args):
        super().__init__()
        self.conv = MP.Conv1d(h_dim, h_dim, *args, groups=h_dim)

    def forward(self, x: Float[Tensor, "B H F L"]) -> Float[Tensor, "B H _F L"]:
        b = x.size(0)
        x = rearrange(x, 'b h f l -> (b l) h f')
        x = self.conv(x)
        return rearrange(x, '(b l) h f -> b h f l', b=b)

class tConv(nn.Module):
    def __init__(self, h_dim, *args):
        super().__init__()
        self.conv = MP.Conv1d(h_dim, h_dim, *args, groups=h_dim)

    def forward(self, x: Float[Tensor, "B H F L"]) -> Float[Tensor, "B H F _L"]:
        b = x.size(0)
        x = rearrange(x, 'b h f l -> (b f) h l')
        x = self.conv(x)
        return rearrange(x, '(b f) h l -> b h f l', b=b)

class AudioEncoder(nn.Module):
    def __init__(self, out_dim: int, in_dim: int, h_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            Rearrange('b f l -> b 1 f l'),
            MP.Conv2d(1, h_dim, 1),
            tConv(h_dim, 3,1,1),
            MP.Conv2d(h_dim, h_dim, 1),
            MP.SiLU(),
            fConv(h_dim, 6,6,1),
            MP.Conv2d(h_dim, h_dim, 1),
            MP.SiLU(),
            tConv(h_dim, 5,1,2),
            MP.Conv2d(h_dim, h_dim, 1),
            MP.SiLU(),
            fConv(h_dim, 12,12,1),
            Rearrange('b h 1 l -> b h l'),
        )
        self.decoder = nn.Sequential(
            MP.Conv1d(h_dim+in_dim, out_dim, 1),
            MP.Gain(),
        )

    def forward(
        self,
        a: Float[Tensor, "B A L"],
        x: Float[Tensor, "B H pL"],
    ) -> Float[Tensor, "B X L"]:
        x = x[:,:,:a.size(-1)]
        h = self.encoder(a)
        return self.decoder(th.cat([x,h], dim=1))