
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from einops import rearrange

class AttendLabel(nn.Module):
    def __init__(
        self,
        x_dim: int,
        label_dim: int,
        head_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.label_dim = label_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        h_dim = head_dim * num_heads

        self.proj_q = nn.Conv1d(x_dim, h_dim, 1)
        self.proj_kv = nn.Linear(label_dim, 2*h_dim)
        self.proj_out = nn.Conv1d(h_dim, x_dim, 1)

    def forward(
        self,
        x: Float[Tensor, "B X L"],
        c: Float[Tensor, "B {self.label_dim}"],
    ) -> Float[Tensor, "B X L"]:
        
        q = F.normalize(rearrange(self.proj_q(x), 'b (h d) l -> b h d l', h=self.num_heads), dim=2)
        k, v = F.normalize(rearrange(self.proj_kv(c), 'b (kv h d) -> kv b h d', kv=2, h=self.num_heads), dim=3).unbind(dim=0)

        o = th.einsum('b h d l, b h d, b h e -> b h e l', q, k, v)

        return self.proj_out(rearrange(o, 'b h e l -> b (h e) l'))