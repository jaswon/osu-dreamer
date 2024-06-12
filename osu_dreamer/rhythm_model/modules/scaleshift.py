
from jaxtyping import Float

from torch import nn, Tensor

class ScaleShift(nn.Module):
    def __init__(self, dim: int, cond_dim: int, net: nn.Module):
        super().__init__()
        self.net = net

        self.norm = nn.GroupNorm(1, dim, affine=False)
        self.to_scale_shift = nn.Linear(cond_dim, dim * 2)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: Float[Tensor, "B D L"], e: Float[Tensor, "B T"]):
        scale, shift = self.to_scale_shift(e).unsqueeze(-1).chunk(2, dim=1)
        return self.net(self.norm(x) * (1+scale) + shift)