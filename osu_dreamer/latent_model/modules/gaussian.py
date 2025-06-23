
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

class GaussianVariationalBottleneck(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mu = nn.Conv1d(dim, dim, 1)
        self.logvar = nn.Conv1d(dim, dim, 1)
 
    def forward(
        self, 
        z: Float[Tensor, "B D L"],
    ) -> tuple[
        Float[Tensor, "B D L"], # quantized
        Float[Tensor, ""],      # loss
    ]:
        mu, logvar = self.mu(z), self.logvar(z)
        std = (0.5 * logvar).exp()
        z = mu + std * th.randn_like(std)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        return z, kl.mean()