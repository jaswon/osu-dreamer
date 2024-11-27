
from dataclasses import dataclass

from jaxtyping import Float

from torch import nn, Tensor 

from osu_dreamer.data.load_audio import A_DIM

import osu_dreamer.modules.mp as MP

@dataclass
class AudioFeatureArgs:
    dim: int
    scales: list[int]
    conv_expand: int
    seq_depth: int
    seq_expand: int

class AudioFeatures(nn.Module):
    def __init__(
        self,
        args: AudioFeatureArgs,
    ):
        super().__init__()

        in_dim = args.dim // 2**len(args.scales)
        assert 2**len(args.scales) * in_dim == args.dim

        self.proj_in = nn.Sequential(
            nn.Unflatten(1, (1, -1)), 
            nn.Conv2d(1, in_dim, 1),
        )

        class conv_layer(nn.Module):
            def __init__(self, d: int, s: int):
                super().__init__()
                h = d*args.conv_expand
                self.res = nn.Sequential(
                    MP.SiLU(),
                    MP.Conv2d(d, h, 1),
                    MP.Conv2d(h, h, *zip((5,1,2), (3,1,1)), groups=h),
                    MP.SiLU(),
                    MP.Conv2d(h, d, 1),
                )
                self.down = nn.Sequential(
                    nn.AvgPool2d((s,1),(s,1)),
                    MP.Conv2d(d, d*2, 1),
                )
        
            def forward(self, x: Float[Tensor, "B d F L"]) -> Float[Tensor, "B D f L"]:
                x = MP.pixel_norm(x)
                h = MP.add(x, self.res(x), t=.3)
                return self.down(h)
            
        self.conv_layers = nn.ModuleList()
        size = 1
        d = in_dim
        for s in args.scales:
            self.conv_layers.append(conv_layer(d, s))
            size *= s
            d *= 2
        assert A_DIM == size

        H = d * args.seq_expand
        class seq_layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.hg = MP.Conv1d(d, H*2, 1)
                self.net = nn.Sequential(
                    MP.SiLU(),
                    MP.Conv1d(H, H, 3,1,1, groups=H),
                    MP.SiLU(),
                    MP.minGRU2(H),
                )
                self.out = MP.Conv1d(H, d, 1)

            def forward(self, x: Float[Tensor, "B X L"]) -> Float[Tensor, "B X L"]:
                x = MP.pixel_norm(x)
                h,g = self.hg(x).chunk(2, dim=1)
                h = self.net(h) * MP.silu(g)
                return MP.add(x, self.out(h), t=.3)
            
        self.seq_layers = nn.ModuleList([ seq_layer() for _ in range(args.seq_depth) ])

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
    ) -> Float[Tensor, "B D L"]:
        h = self.proj_in(audio)
        for conv_layer in self.conv_layers:
            h = conv_layer(h)
        h = h.flatten(2)
        for seq_layer in self.seq_layers:
            h = seq_layer(h)
        return MP.silu(h)