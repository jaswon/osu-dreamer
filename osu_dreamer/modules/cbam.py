from jaxtyping import Float

import torch as th
import torch.nn.functional as F
from torch import nn, Tensor

class CBAM(nn.Module):
    def __init__(
        self,
        dim: int,
        reduction: int = 8,
    ):
        super().__init__()

        self.ch_map = nn.Sequential(
            nn.Linear(dim, dim//reduction),
            nn.SiLU(),
            nn.Linear(dim//reduction, dim),
        )

        self.sp_map = nn.Conv1d(2,1,7,1,3)

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L"]:
        ch_attn = self.ch_map(th.stack([
            x.max(dim=2).values, 
            x.mean(dim=2),
        ])).sum(dim=0)[...,None]
        x = x * F.tanh(ch_attn)

        sp_attn = self.sp_map(th.cat([
            x.max(dim=1, keepdim=True).values,
            x.mean(dim=1, keepdim=True),
        ], dim=1))
        x = x * F.tanh(sp_attn)

        return x