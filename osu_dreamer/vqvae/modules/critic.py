
from jaxtyping import Float

from torch import nn, Tensor

import osu_dreamer.modules.mp as MP

CriticArgs = list[tuple[int, int, int, int]] # out_dim, kernel_size, stride, group

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