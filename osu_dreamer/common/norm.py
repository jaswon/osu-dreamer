
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(th.ones(dim,1))

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L"]:
        return F.normalize(x, dim = 1, eps=1e-8) * self.gamma * self.scale