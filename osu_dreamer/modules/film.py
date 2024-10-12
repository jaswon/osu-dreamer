
from jaxtyping import Float

from torch import nn, Tensor

class FiLM(nn.Module):
    def __init__(self, dim: int, t_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(dim, dim, affine=False)
        self.ss = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, dim*2),
        )

    def forward(self, xt: tuple[
        Float[Tensor, "B D ..."], 
        Float[Tensor, "B T"],
    ]) -> Float[Tensor, "B D ..."]:
        x,t = xt
        ss = self.ss(t)
        ss = ss.view(*ss.shape, *([1] * (x.ndim - 2)))
        scale, shift = ss.chunk(2, dim=1)
        return self.norm(x) * scale + shift