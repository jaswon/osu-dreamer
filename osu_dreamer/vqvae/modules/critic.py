
from jaxtyping import Float

from torch import nn, Tensor

import osu_dreamer.modules.mp as MP

class Critic(nn.Module):
    def __init__(
        self,
        dim: int,
        config: list[tuple[int, int, int]], # dim, kernel_size, stride
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        i = dim
        for o,k,s in config:
            self.convs.append(nn.Sequential(
                MP.Conv1d(i, i, k, s, groups=i),
                MP.Conv1d(i, o, 1)
            ))
            i = o
        self.convs.append(MP.Conv1d(i, 1, 1))

    def forward(self, x: Float[Tensor, "B D L"]) -> list[Float[Tensor, "B _D _L"]]:
        f_maps = [x]
        for conv in self.convs:
            f_maps.append(conv(f_maps[-1]))
        return f_maps
        