
from jaxtyping import Float

from torch import nn, Tensor

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.spec_features import SpecFeatures

class FiLM(nn.Module):
    def __init__(self, dim: int, h_dim: int):
        super().__init__()
        self.gamma = nn.Sequential(MP.Conv1d(h_dim, dim, 1), MP.Gain())
        self.beta = nn.Sequential(MP.Conv1d(h_dim, dim, 1), MP.Gain())

    def forward(
        self,
        a: Float[Tensor, "*B H L"],
        x: Float[Tensor, "B X pL"],
    ) -> Float[Tensor, "B X L"]:
        return self.beta(a) + (1+self.gamma(a)) * x[:,:,:a.size(-1)]

class AudioEncoder(nn.Module):
    def __init__(self, out_dim: int, in_dim: int, h_dim: int):
        super().__init__()
        self.encoder = SpecFeatures(h_dim)
        self.film = FiLM(in_dim, h_dim)
        self.decoder = nn.Sequential(
            MP.Conv1d(in_dim, in_dim, 5,1,2, groups=in_dim),
            MP.Conv1d(in_dim, out_dim, 1),
            MP.Gain(),
        )

    def forward(
        self,
        a: Float[Tensor, "*B A L"],
        x: Float[Tensor, "B H pL"],
    ) -> Float[Tensor, "B X L"]:
        x = self.film(self.encoder(a), x)
        return self.decoder(x)