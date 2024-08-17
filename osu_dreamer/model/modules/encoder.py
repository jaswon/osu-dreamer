
from dataclasses import dataclass

from jaxtyping import Float, Int

from torch import nn, Tensor

from osu_dreamer.common.residual import ResStack

@dataclass
class EncoderArgs:
    stack_depth: int

class Encoder(nn.Module):
    def __init__(self, dim: int, args: EncoderArgs, in_dim: int = 0):
        super().__init__()
        self.proj_in = nn.Identity() if in_dim==0 else nn.Conv1d(in_dim, dim, 1)

        self.net = ResStack(dim, [
            nn.Conv1d(dim, dim, 5,1,2, groups=dim)
            for _ in range(args.stack_depth)
        ])

    def forward(
        self,
        a: Float[Tensor, "B A L"],
        p: Int[Tensor, "B L"],
    ) -> Float[Tensor, "B H L"]:
        return self.net(self.proj_in(a))