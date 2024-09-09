
from jaxtyping import Float

from torch import nn, Tensor

class ScaleShift(nn.Module):
    def __init__(self, dim: int, t_dim: int, net: nn.Module):
        super().__init__()
        self.net = net

        self.norm = nn.GroupNorm(dim, dim, affine=False)
        self.ss = nn.Linear(t_dim, dim*2)
        nn.init.zeros_(self.ss.weight)
        nn.init.zeros_(self.ss.bias)

    def forward(self, x: Float[Tensor, "B X L"], t: Float[Tensor, "B T"]):
        scale, shift = self.ss(t)[...,None].chunk(2, dim=1)
        return self.net(self.norm(x) * (1+scale) + shift)