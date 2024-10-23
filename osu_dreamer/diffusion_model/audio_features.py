
from dataclasses import dataclass

from jaxtyping import Float

from torch import nn, Tensor 

from osu_dreamer.data.load_audio import A_DIM

from osu_dreamer.modules.mingru import minGRU2

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

        self.net = nn.Sequential(
            nn.Unflatten(1, (1, -1)), 
            nn.Conv2d(1, in_dim, 1),
        )

        size = 1
        d = in_dim
        for s in args.scales:
            self.net.extend([
                nn.Conv2d(d, d, (1,9), 1, (0,4), (1,1), groups=d),
                nn.SiLU(),
                nn.Conv2d(d, d*2, 1),
                minGRU2(),
                nn.MaxPool2d((s,1), (s,1)),
                nn.Conv2d(d, d*2, 1),
            ])
            size *= s
            d *= 2
        assert A_DIM == size

        self.net.append(nn.Flatten(1,2))

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
    ) -> Float[Tensor, "B D L"]:
        return self.net(audio)