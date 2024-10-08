
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.data.prepare_map import NUM_LABELS

from osu_dreamer.modules.cbam import CBAM
from osu_dreamer.modules.rff import RandomFourierFeatures
from osu_dreamer.modules.film import FiLM
from osu_dreamer.modules.wavenet import WaveNet, WaveNetArgs
    
@dataclass
class DenoiserArgs:
    h_dim: int
    cbam_reduction: int

    c_features: int
    c_dim: int

    wavenet_args: WaveNetArgs

class Denoiser(nn.Module):
    def __init__(
        self,
        dim: int,
        a_dim: int,
        args: DenoiserArgs,
    ):
        super().__init__()
        
        self.proj_c = nn.Sequential(
            RandomFourierFeatures(1 + NUM_LABELS, args.c_features),
            nn.Linear(args.c_features, args.c_dim),
            nn.SiLU(),
        )

        self.proj_h = nn.Conv1d(a_dim + dim, args.h_dim, 1)

        class DenoiserBlock(nn.Module):
            def __init__(self, dim: int):
                super().__init__()
                self.norm1 = FiLM(dim, args.c_dim)
                self.conv = nn.Sequential( nn.SiLU(), nn.Conv1d(dim, dim, 1) )
                self.norm2 = FiLM(dim, args.c_dim)
                self.attn = nn.Sequential( nn.SiLU(), CBAM(dim, args.cbam_reduction) )

            def forward(
                self, 
                x: Float[Tensor, "B D L"], 
                t: Float[Tensor, "B T"],
            ) -> Float[Tensor, "B D L"]:
                x = x + self.conv(self.norm1(x, t))
                return x + self.attn(self.norm2(x, t))

        self.net = WaveNet(args.h_dim, args.wavenet_args, DenoiserBlock)

        self.proj_out = nn.Conv1d(args.h_dim, dim, 1)
        th.nn.init.zeros_(self.proj_out.weight)
        th.nn.init.zeros_(self.proj_out.bias) # type: ignore

    def forward(
        self, 
        audio_features: Float[Tensor, "B A L"],
        label: Float[Tensor, str(f"B {NUM_LABELS}")],
        
        # --- diffusion args --- #
        x: Float[Tensor, "B X L"],  # noised input
        t: Float[Tensor, "B"],      # (log) denoising step
    ) -> Float[Tensor, "B X L"]:
        c = self.proj_c(th.cat([t[:,None], label], dim=1))
        h = self.proj_h(th.cat([ audio_features, x ], dim=1))
        return self.proj_out(self.net(h,c))