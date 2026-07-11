
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

def rms_norm(
    x: Float[Tensor, "B C *N"], 
    gamma: Float[Tensor, "C"] | None = None,
) -> Float[Tensor, "B C *N"]:
    xf = x.float()
    inv_rms = xf.pow(2).mean(dim=1, keepdim=True).add(1e-6).rsqrt()
    normed = (xf * inv_rms).to(x.dtype)
    if gamma is not None:
        normed = normed * gamma[:,*((None,) * (x.ndim-2))]
    return normed

class RMSNorm(nn.Module):
    def __init__(self, dim: int, affine: bool = True, gain: float = 1.):
        super().__init__()
        self.gamma = None
        if affine:
            self.gamma = nn.Parameter(th.full((dim,), float(gain)))

    def forward(self, x: Float[Tensor, "B C *N"]) -> Float[Tensor, "B C *N"]:
        return rms_norm(x, self.gamma)