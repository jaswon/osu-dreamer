
from jaxtyping import Float

from torch import nn, Tensor

from einops.layers.torch import Rearrange

class SimpleAudioEncoder(nn.Module):
    def __init__(self, h_dim: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            Rearrange('b f l -> b 1 f l'),
            nn.Conv2d(1, 1, (1, 7), (1,1), (1,3)),
            nn.SiLU(),
            nn.AdaptiveMaxPool2d((5, None)), # b 1 5 l
            Rearrange('b 1 d l -> b d l'),
            nn.Conv1d(5, h_dim, 1),
            Rearrange('b h l -> b l h'),
        )

    def forward(self, x: Float[Tensor, "B A L"]) -> Float[Tensor, "B L D"]:
        return self.encoder(x)
