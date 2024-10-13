
from dataclasses import dataclass

from jaxtyping import Float

from torch import nn, Tensor

from einops import rearrange

from osu_dreamer.data.load_audio import A_DIM

class Residual(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return x + self.net(x)

@dataclass
class AudioFeatureArgs:
    scales: list[int]

class AudioFeatures(nn.Module):
    def __init__(
        self,
        dim: int,
        args: AudioFeatureArgs,
    ):
        super().__init__()

        in_dim = dim // 2**len(args.scales)
        assert 2**len(args.scales) * in_dim == dim

        self.proj_in = nn.Conv2d(1, in_dim, 1)
        self.blocks = nn.ModuleList()
        size = 1
        d = in_dim
        for s in args.scales:
            self.blocks.append(nn.Sequential(
                Residual(nn.Conv2d(d, d, (1,3), 1, (0,1), (1,1), groups=d)),
                Residual(nn.Conv2d(d, d, (1,5), 1, (0,6), (1,3), groups=d)),
                nn.Conv2d(d, d*2, 1),
                nn.GLU(dim=1),
                nn.MaxPool2d((s,1), (s,1)),
                nn.Conv2d(d, d*2, 1),
            ))
            size *= s
            d *= 2

        assert A_DIM == size

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
    ) -> Float[Tensor, "B D L"]:
        h = self.proj_in(rearrange(audio, 'b a l -> b 1 a l'))
        for block in self.blocks:
            h = block(h)
        return rearrange(h, 'b d 1 l -> b d l')