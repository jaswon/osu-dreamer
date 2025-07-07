
from jaxtyping import Float

from dataclasses import dataclass

from torch import nn, Tensor
import torch.nn.functional as F

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.spec_features import SpecFeatures

from .audio import FiLM

CriticArgs = list[tuple[int, int, int, int]] # out_dim, kernel_size, stride, group

@dataclass
class MultiScaleCriticArgs:
    spec_features: int
    scales: int
    convs: CriticArgs

class MultiScaleCritic(nn.Module):
    def __init__(
        self,
        dim: int,
        args: MultiScaleCriticArgs,
    ):
        super().__init__()
        self.encoder = SpecFeatures(args.spec_features)
        self.film = FiLM(dim, args.spec_features)
        self.critics = nn.ModuleList([
            Critic(dim, args.convs) 
            for _ in range(args.scales)
        ])

    def forward(
        self, 
        a: Float[Tensor, "B A L"],
        x: Float[Tensor, "B D L"],
    ) -> list[list[Float[Tensor, "B _D _L"]]]:
        fmaps = []
        for i, critic in enumerate(self.critics):
            if i == 0:
                x = self.film(self.encoder(a), x)
            else:
                x = F.avg_pool1d(x, 4,2,1)
            fmaps.append(critic(x))
        return fmaps

class Critic(nn.Module):
    def __init__(
        self,
        dim: int,
        config: CriticArgs, 
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        i = dim
        for o,k,s,g in config:
            self.convs.append(nn.Sequential(
                MP.Conv1d(i, o, k, s, groups=g),
                MP.SiLU(),
            ))
            i = o
        self.convs.append(MP.Conv1d(i, 1, 1))

    def forward(self, x: Float[Tensor, "B D L"]) -> list[Float[Tensor, "B _D _L"]]:
        f_maps = [x]
        for conv in self.convs:
            f_maps.append(conv(f_maps[-1]))
        return f_maps[1:]