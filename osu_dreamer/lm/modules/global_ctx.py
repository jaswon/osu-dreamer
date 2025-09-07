
from jaxtyping import Float

from torch import nn, Tensor

from einops.layers.torch import Rearrange

class GlobalEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num: int):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(in_dim, in_dim, 7, 1, 3, groups=in_dim),
            nn.Conv1d(in_dim, in_dim, 1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_dim, out_dim * num, 1),
            Rearrange('b (n d) 1 -> b n d', n=num),
        )


    def forward(self, x: Float[Tensor, "1 iD L"]) -> Float[Tensor, "G oD"]:
        return self.conv(x)[0]