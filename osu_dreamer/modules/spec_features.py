
from jaxtyping import Float

from torch import nn, Tensor

from einops.layers.torch import Rearrange
from einops import rearrange

import osu_dreamer.modules.mp as MP

class sepConv(nn.Module):
    def __init__(self, shape: str, h_dim: int, *args):
        super().__init__()
        self.shape = shape
        self.conv = MP.Conv1d(h_dim, h_dim, *args, groups=h_dim)

    def forward(self, x: Float[Tensor, "B H F L"]) -> Float[Tensor, "B H _F L"]:
        b = x.size(0)
        x = rearrange(x, f'b h f l -> {self.shape}')
        x = self.conv(x)
        return rearrange(x, f'{self.shape} -> b h f l', b=b)

class SpecFeatures(nn.Sequential):
    def __init__(self, h_dim: int):
        super().__init__(
            Rearrange('b f l -> b 1 f l'),
            MP.Conv2d(1, h_dim, 1),
            *(
                layer for t,f in [
                    (3,3),
                    (5,4),
                    (9,6),
                ]
                for conv in [
                    sepConv('(b f) h l', h_dim, t,1,t//2), 
                    sepConv('(b l) h f', h_dim, f,f,1),
                ]
                for layer in [
                    conv, 
                    MP.Conv2d(h_dim, h_dim, 1),
                    MP.SiLU(),
                ]
            ),
            Rearrange('b h 1 l -> b h l'),
        )