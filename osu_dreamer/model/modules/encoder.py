
from dataclasses import dataclass

from jaxtyping import Float

from torch import nn, Tensor

from .wavenet import WaveNet, WaveNetArgs

@dataclass
class EncoderArgs(WaveNetArgs):
    pass

class Encoder(nn.Module):
    def __init__(self, dim: int, args: EncoderArgs, in_dim: int = 0):
        super().__init__()
        self.proj_in = nn.Identity() if in_dim==0 else nn.Conv1d(in_dim, dim, 1)
        self.net = WaveNet(dim, args)

    def forward(
        self,
        a: Float[Tensor, "B A L"],
    ) -> Float[Tensor, "B H L"]:
        return self.net(self.proj_in(a))