
from dataclasses import dataclass

from jaxtyping import Float

from torch import nn, Tensor 

from osu_dreamer.data.load_audio import A_DIM

from osu_dreamer.modules.mingru import minGRU2

@dataclass
class AudioFeatureArgs:
    scales: list[int]
    conv_expand: int
    seq_depth: int
    seq_expand: int

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
            h = d*args.conv_expand
            self.net.extend([
                nn.Conv2d(d, h, 1),
                nn.Conv2d(h, h, *zip((5,1,2), (3,1,1)), groups=h),
                nn.SiLU(),
                nn.Conv2d(h, d, 1),
                nn.ReLU(),
                nn.MaxPool2d((s,1), (s,1)),
                nn.Conv2d(d, d*2, 1),
            ])
            size *= s
            d *= 2
        assert A_DIM == size

        self.net.append(nn.Flatten(1,2))

        class layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.GroupNorm(1, d),
                    nn.Conv1d(d, d, 3,1,1, groups=d),
                    nn.SiLU(),
                    nn.Conv1d(d, d*args.seq_expand*2, 1),
                    minGRU2(),
                    nn.Conv1d(d*args.seq_expand, d, 1),
                )

            def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L"]:
                return x + self.net(x)
            
        for _ in range(args.seq_depth):
            self.net.append(layer())

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
    ) -> Float[Tensor, "B D L"]:
        return self.net(audio)