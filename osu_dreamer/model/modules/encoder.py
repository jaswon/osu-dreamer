
from dataclasses import dataclass

from jaxtyping import Float

from torch import nn, Tensor

from .residual import ResStack

class WaveNet(ResStack):
    """receptive field = 1+s*(2**d-1)"""
    def __init__(self, dim: int, num_stacks: int, stack_depth: int):
        super().__init__(dim, [
            nn.Sequential(
                nn.ZeroPad1d((1,0) if d==0 else 2**(d-1)),
                nn.Conv1d(dim, dim, 2, dilation=2**d),
                nn.SiLU(),
            )
            for _ in range(num_stacks)
            for d in range(stack_depth)
        ]) 

@dataclass
class EncoderArgs:
    num_stacks: int
    stack_depth: int

class Encoder(nn.Module):
    def __init__(self, dim: int, args: EncoderArgs, in_dim: int = 0):
        super().__init__()
        self.proj_in = nn.Identity() if in_dim==0 else nn.Conv1d(in_dim, dim, 1)
        self.net = WaveNet(dim, args.num_stacks, args.stack_depth)

    def forward(
        self,
        a: Float[Tensor, "B A L"],
    ) -> Float[Tensor, "B H L"]:
        return self.net(self.proj_in(a))