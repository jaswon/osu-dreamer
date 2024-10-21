
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from osu_dreamer.data.prepare_map import NUM_LABELS

from osu_dreamer.modules.cbam import CBAM
from osu_dreamer.modules.modconv import ModulatedConv1d
from osu_dreamer.modules.wavenet import WaveNet, WaveNetArgs
    
@dataclass
class DenoiserArgs:
    h_dim: int
    cbam_reduction: int

    c_dim: int
    c_depth: int

    wavenet_args: WaveNetArgs

class Denoiser(nn.Module):
    def __init__(
        self,
        dim: int,
        a_dim: int,
        args: DenoiserArgs,
    ):
        super().__init__()
        
        self.proj_c = nn.Sequential(*(
            block for i in range(args.c_depth)
            for block in [
                nn.Linear(1+NUM_LABELS if i==0 else args.c_dim, args.c_dim),
                nn.SiLU(),
            ]
        ))

        self.proj_h = ModulatedConv1d(a_dim+dim, args.h_dim, args.c_dim)

        class DenoiserBlock(nn.Module):
            def __init__(self, depth: int):
                super().__init__()
                self.attn = None
                if depth >= 4:
                    self.attn = CBAM(args.h_dim, args.cbam_reduction)

                self.c1 = ModulatedConv1d(args.h_dim, args.h_dim, args.c_dim)
                self.c2 = ModulatedConv1d(args.h_dim, args.h_dim, args.c_dim)

            def forward(
                self, 
                x: Float[Tensor, "B D L"], 
                t: Float[Tensor, "B T"],
            ) -> Float[Tensor, "B D L"]:
                if self.attn is not None:
                    x = x + self.attn(x)
                x = x + self.c2((F.silu(self.c1((x,t))),t))
                return x

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
        c = self.proj_c(th.cat([t[:,None],label], dim=1))
        h = self.proj_h((th.cat([audio_features,x], dim=1), c))
        return self.proj_out(self.net(h,c))