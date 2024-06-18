
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

# Split/Join via projection

class ProjSplit(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Conv1d(dim, 2*dim, 1)

    def forward(self, x: Float[Tensor, "B X L"]) -> tuple[Float[Tensor, "B X L"], Float[Tensor, "B X L"]]:
        return self.net(x).chunk(2, dim=1)

class ProjJoin(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Conv1d(2*dim, dim, 1)

    def forward(self, h: Float[Tensor, "B X L"], x: Float[Tensor, "B X L"]) -> Float[Tensor, "B X L"]:
        return self.net(th.cat([h, x], dim=1))
    

# Split/Join via residual

class ResSplit(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Conv1d(dim, dim, 1)

    def forward(self, x: Float[Tensor, "B X L"]) -> tuple[Float[Tensor, "B X L"], Float[Tensor, "B X L"]]:
        h = self.net(x)
        return h, h-x
    
class ResJoin(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Conv1d(dim, dim, 1)

    def forward(self, h: Float[Tensor, "B X L"], x: Float[Tensor, "B X L"]) -> Float[Tensor, "B X L"]:
        return self.net(x+h)
    
Split, Join = ProjSplit, ProjJoin
# Split, Join = ResSplit, ResJoin