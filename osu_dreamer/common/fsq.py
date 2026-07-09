
from jaxtyping import Float

import torch as th
from torch import Tensor, nn

def round_ste(x: Tensor) -> Tensor:
    return x + (x.round() - x).detach()

def bound(x: Tensor) -> Tensor:
    # return x.tanh() # fsq
    return th.sigmoid(x*1.6)*2-1 # improved fsq

class FSQ(nn.Module):
    def __init__(self, levels: int):
        super().__init__()
        assert levels % 2 == 1, "FSQ: levels must be odd"
        self.half_l = (levels-1)/2

    def forward(self, z: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L"]:
        return round_ste(bound(z) * self.half_l) / self.half_l