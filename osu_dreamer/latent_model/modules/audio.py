
from jaxtyping import Float

import torch as th
from torch import nn, Tensor


import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.spec_features import SpecFeatures

class AudioEncoder(nn.Module):
    def __init__(self, out_dim: int, in_dim: int, h_dim: int):
        super().__init__()
        self.encoder = SpecFeatures(h_dim)
        self.decoder = nn.Sequential(
            MP.Conv1d(h_dim+in_dim, out_dim, 1),
            MP.Gain(),
        )

    def forward(
        self,
        a: Float[Tensor, "*B A L"],
        x: Float[Tensor, "B H pL"],
    ) -> Float[Tensor, "B X L"]:
        x = x[:,:,:a.size(-1)]
        h = self.encoder(a).expand(x.size(0),-1,-1)
        return self.decoder(th.cat([x,h], dim=1))