
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

class WaveNet(nn.Module):
    """receptive field = 1+s*(2**d-1))"""

    def __init__(self, dim: int, num_stacks: int, stack_depth: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.ZeroPad1d((1,0) if d==0 else 2**(d-1)),
                nn.Conv1d(dim, 2*dim, 2, dilation=2**d),
                nn.GLU(dim=1),
                nn.Conv1d(dim, 2*dim, 1),
            )
            for _ in range(num_stacks)
            for d in range(stack_depth)
        ])

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L"]:
        o = th.zeros_like(x)
        for block in self.blocks:
            res, skip = block(x).chunk(2, dim=1)
            x = (x + res) * 2 ** -0.5
            o = o + skip
        return o * len(self.blocks) ** -0.5