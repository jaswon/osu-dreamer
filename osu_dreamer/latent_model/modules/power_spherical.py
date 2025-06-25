from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from einops import rearrange

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.power_spherical import PowerSpherical, HypersphericalUniform

def normalize(x, eps=1e-8):
    return x / (th.linalg.vector_norm(x, dim=1, keepdim=True) + eps)

class PowerSphericalVariationalBottleneck(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.loc = MP.Conv1d(dim, dim, 1)
        self.log_scale = nn.Conv1d(dim, 1, 1)

    def forward(
        self,
        z: Float[Tensor, "B D L"],
    ) -> tuple[
        Float[Tensor, "B D L"], # quantized
        Float[Tensor, ""],      # loss
    ]:        
        B,D,_ = z.size()
        loc_flat = rearrange(normalize(self.loc(z)), 'b d l -> (b l) d')
        scale_flat = rearrange(self.log_scale(z).exp(), 'b 1 l -> (b l)')

        posterior = PowerSpherical(loc_flat, scale_flat)
        z = rearrange(posterior.rsample(), '(b l) d -> b d l', b=B)
        
        prior = HypersphericalUniform(D, device=z.device, dtype=z.dtype) # type: ignore
        kl_loss = th.distributions.kl_divergence(posterior, prior).mean()
        
        return z, kl_loss