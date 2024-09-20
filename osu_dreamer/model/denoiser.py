
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.data.beatmap.encode import X_DIM
from osu_dreamer.data.prepare_map import NUM_LABELS

from .modules.unet import UNet
from .modules.cbam import CBAM
from .modules.rff import RandomFourierFeatures

class VarSequential(nn.Sequential):
    def forward(self, x, *args, **kwargs):
        for module in self:
            x = module(x, *args, **kwargs)
        return x
    
class AdaGN(nn.Module):
    def __init__(self, dim: int, t_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(dim, dim, affine=False)
        self.ss = nn.Linear(t_dim, dim*2)
        nn.init.zeros_(self.ss.weight)
        nn.init.zeros_(self.ss.bias)

    def forward(
        self, 
        x: Float[Tensor, "B D L"], 
        t: Float[Tensor, "B T"],
    ) -> Float[Tensor, "B D L"]:
        scale, shift = self.ss(t)[...,None].chunk(2, dim=1)
        return self.norm(x) * (1+scale) + shift

@dataclass
class DenoiserArgs:
    h_dim: int
    cbam_reduction: int

    c_features: int
    c_dim: int

    scales: list[int]
    stack_depth: int
    block_depth: int
    expand: int

class Denoiser(nn.Module):
    def __init__(
        self,
        a_dim: int,
        args: DenoiserArgs,
    ):
        super().__init__()
        
        self.proj_c = nn.Sequential(
            RandomFourierFeatures(1 + NUM_LABELS, args.c_features),
            nn.Linear(args.c_features, args.c_dim),
            nn.SiLU(),
        )

        self.proj_h = nn.Conv1d(a_dim + X_DIM, args.h_dim, 1)

        hh_dim = args.h_dim * args.expand
        class DenoiserUNetBlock(nn.Module):
            def __init__(self, net: nn.Module):
                super().__init__()
                self.proj_in = nn.Conv1d(args.h_dim, hh_dim, 1)
                self.norm = AdaGN(hh_dim, args.c_dim)
                self.proj_out = nn.Sequential(
                    nn.SiLU(),
                    net,
                    nn.Conv1d(hh_dim, args.h_dim, 1),
                    nn.SiLU(),
                )

            def forward(
                self, 
                x: Float[Tensor, "B D L"], 
                t: Float[Tensor, "B T"],
            ) -> Float[Tensor, "B D L"]:
                h = self.norm(self.proj_in(x), t)
                return x + self.proj_out(h)

        self.net = UNet(
            args.h_dim, args.scales,
            VarSequential(*(
                DenoiserUNetBlock(CBAM(hh_dim, args.cbam_reduction))
                for _ in range(args.stack_depth)
            )),
            lambda: VarSequential(*(
                DenoiserUNetBlock(nn.Conv1d(hh_dim, hh_dim, 5,1,2, groups=hh_dim))
                for _ in range(args.block_depth)
            )),
        )

        self.proj_out = nn.Conv1d(args.h_dim, X_DIM, 1)
        th.nn.init.zeros_(self.proj_out.weight)
        th.nn.init.zeros_(self.proj_out.bias) # type: ignore

    def forward(
        self, 
        audio_features: Float[Tensor, "B A L"],
        label: Float[Tensor, str(f"B {NUM_LABELS}")],
        
        # --- diffusion args --- #
        x: Float[Tensor, str(f"B {X_DIM} L")],  # noised input
        t: Float[Tensor, "B"],                  # (log) denoising step
    ) -> Float[Tensor, str(f"B {X_DIM} L")]:
        c = self.proj_c(th.cat([t[:,None], label], dim=1))
        h = self.proj_h(th.cat([ audio_features, x ], dim=1))
        return self.proj_out(self.net(h,c))