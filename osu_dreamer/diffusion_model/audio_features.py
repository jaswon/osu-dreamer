
from dataclasses import dataclass

from jaxtyping import Float

import torch.nn.functional as F
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

        self.conv = nn.Sequential(
            nn.Unflatten(1, (1, -1)), 
            nn.Conv2d(1, in_dim, 1, bias=False),
        )

        size = 1
        d = in_dim
        for s in args.scales:
            h = d*args.conv_expand
            self.conv.extend([
                nn.Conv2d(d, h, 1, bias=False),
                nn.Conv2d(h, h, *zip((5,1,2), (3,1,1)), groups=h, bias=False),
                nn.SiLU(),
                nn.Conv2d(h, d, 1, bias=False),
                nn.ReLU(),
                nn.MaxPool2d((s,1), (s,1)),
                nn.Conv2d(d, d*2, 1, bias=False),
            ])
            size *= s
            d *= 2
        assert A_DIM == size

        self.conv.append(nn.Flatten(1,2))

        H = d * args.seq_expand
        class layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.hg = nn.Conv1d(d, H*2, 1, bias=False)
                self.net = nn.Sequential(
                    nn.Conv1d(H, H, 3,1,1, groups=H, bias=False),
                    nn.SiLU(),
                    nn.Conv1d(H, H*2, 1, bias=False),
                    minGRU2(),
                )
                self.out = nn.Conv1d(H, d, 1, bias=False)

            def forward(self, x: Float[Tensor, "B X L"]) -> Float[Tensor, "B X L"]:
                h,g = self.hg(x).chunk(2, dim=1)
                h = self.net(h) * F.silu(g)
                return x + self.out(h)
            
        self.layers = nn.ModuleList([ layer() for _ in range(args.seq_depth) ])

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
    ) -> Float[Tensor, "B D L"]:
        h = self.conv(audio)
        for layer in self.layers:
            h = layer(h)
        return F.silu(h)