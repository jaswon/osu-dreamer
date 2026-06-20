
from jaxtyping import Float

import torch as th
from torch import nn, Tensor


class Derf(nn.Module):
    def __init__(
        self,
        *shape: int,
        alpha_init: float = .5,
        learn_affine: bool = True,
    ):
        super().__init__()
        self.scale = nn.Parameter(th.ones(1) * alpha_init)
        self.shift = nn.Parameter(th.zeros(1))
        self.learn_affine = learn_affine
        if learn_affine:
            self.gamma = nn.Parameter(th.ones(shape))
            self.beta = nn.Parameter(th.zeros(shape))

    def forward(self, x: Float[Tensor, "*S"]) -> Float[Tensor, "*S"]:
        y = th.erf(self.scale * x + self.shift)
        if self.learn_affine:
            y = self.gamma * y + self.beta
        return y