
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

    def forward(
        self, 
        x: Float[Tensor, "B D L"], 
        t: Float[Tensor, "B T"],
    ) -> Float[Tensor, "B D L"]:
        scale, shift = self.ss(t)[:,:,None].chunk(2, dim=1)
        return self.norm(x) * scale + shift