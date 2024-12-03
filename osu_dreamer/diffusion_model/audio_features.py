
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
from torch import nn, Tensor 

from osu_dreamer.data.load_audio import A_DIM

import osu_dreamer.modules.mp as MP

@dataclass
class AudioFeatureArgs:
    dim: int
    depth: int
    expand: int

class AudioFeatures(nn.Module):
    def __init__(
        self,
        args: AudioFeatureArgs,
    ):
        super().__init__()

        self.proj_in = MP.Conv1d(A_DIM+1, args.dim, 1)

        H = args.dim * args.expand
        class layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.h = nn.Sequential(
                    MP.SiLU(),
                    MP.Conv1d(args.dim, H, 1),
                    MP.Conv1d(H, H, 3,1,1, groups=H),
                    MP.SiLU(),
                    MP.minGRU2(H),
                    MP.PixelNorm(),
                )
                self.g = nn.Sequential(
                    MP.SiLU(),
                    MP.Conv1d(args.dim, H, 1),
                    MP.SiLU(),
                )
                self.out = MP.Conv1d(H, args.dim, 1)

            def forward(self, x: Float[Tensor, "B X L"]) -> Float[Tensor, "B X L"]:
                x = MP.pixel_norm(x)
                o = self.out(self.h(x) * self.g(x))
                return MP.add(x, o, t=.3)
            
        self.layers = nn.ModuleList([ layer() for _ in range(args.depth) ])

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
    ) -> Float[Tensor, "B D L"]:
        audio = MP.cat([audio, th.ones_like(audio[:,:1,:])], dim=1)
        h = self.proj_in(audio)
        for layer in self.layers:
            h = layer(h)
        return MP.silu(h)