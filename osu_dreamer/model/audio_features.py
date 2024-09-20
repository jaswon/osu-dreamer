
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.data.load_audio import A_DIM

@dataclass
class AudioFeatureArgs:
    num_stacks: int
    stack_depth: int
    expand: int

class AudioFeatures(nn.Module):
    def __init__(
        self,
        dim: int,
        args: AudioFeatureArgs,
    ):
        super().__init__()

        self.proj_h = nn.Conv1d(A_DIM, dim, 1)

        # wavenet receptive field: 1+s*(2**d-1))
        self.blocks = nn.ModuleList()
        for _ in range(args.num_stacks):
            for d in range(args.stack_depth):
                self.blocks.append(nn.Sequential(
                    nn.ZeroPad1d((1,0) if d==0 else 2**(d-1)),
                    nn.Conv1d(dim, 2*dim*args.expand, 2, dilation=2**d),
                    nn.GLU(dim=1),
                    nn.Conv1d(dim*args.expand, 2*dim, 1),
                ))

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
    ) -> Float[Tensor, "B A L"]:
        x = self.proj_h(audio)
        o = th.zeros_like(x)
        for block in self.blocks:
            res, skip = block(x).chunk(2, dim=1)
            x = (x + res) * 2 ** -0.5
            o = o + skip
        return o * len(self.blocks) ** -0.5