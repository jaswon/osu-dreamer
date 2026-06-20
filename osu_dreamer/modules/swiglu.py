
from jaxtyping import Float

from torch import nn, Tensor
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(
        self, 
        dim: int, 
        expand: int = 2, 
        dropout: float = 0.,
        radius: int = 1,
    ):
        super().__init__()
        h_dim = int(dim * expand * 2/3)
        self.proj_vg = nn.Conv1d(dim, 2*h_dim, 1+2*radius,1,radius)
        self.dropout = nn.Dropout1d(dropout)
        self.proj_o = nn.Conv1d(h_dim, dim, 1)

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L"]:
        v, g = self.proj_vg(x).chunk(2, dim=1)
        return self.proj_o(self.dropout(v * F.silu(g)))
