
from jaxtyping import Float

from torch import nn, Tensor

from einops.layers.torch import Rearrange

class GlobalEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, in_dim*num, 5,1,2, groups=in_dim),
            Rearrange('b (d n) l -> b (n d) l'),
            nn.SiLU(),
            nn.Conv1d(in_dim*num, out_dim*num, 1),
            nn.AdaptiveAvgPool1d(1),
            Rearrange('b (n d) 1 -> b n d'),
        )


    def forward(self, x: Float[Tensor, "1 iD L"]) -> Float[Tensor, "G oD"]:
        return self.conv(x)[0]