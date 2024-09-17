
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.data.beatmap.encode import X_DIM
from osu_dreamer.data.load_audio import A_DIM
from osu_dreamer.data.prepare_map import NUM_LABELS

from .modules.wavenet import WaveNet
from .modules.unet import UNet
from .modules.cbam import CBAM
from .modules.rff import RandomFourierFeatures

class VarSequential(nn.Sequential):
    def forward(self, x, *args, **kwargs):
        for module in self:
            x = module(x, *args, **kwargs)
        return x

class DenoiserUNetBlock(nn.Module):
    def __init__(self, dim: int, t_dim: int, expand: int, net: nn.Module):
        super().__init__()
        h_dim = dim * expand
        self.proj_in = nn.Sequential(
            nn.Conv1d(dim, h_dim, 1),
            nn.Conv1d(h_dim, h_dim, 5,1,2, groups=h_dim)
        )

        # AdaGN
        self.norm = nn.GroupNorm(h_dim, h_dim, affine=False)
        self.ss = nn.Linear(t_dim, h_dim*2)
        nn.init.zeros_(self.ss.weight)
        nn.init.zeros_(self.ss.bias)

        self.proj_out = nn.Sequential(
            nn.SiLU(),
            net,
            nn.Conv1d(h_dim, dim, 1)
        )

    def forward(self, x: Float[Tensor, "B D L"], t: Float[Tensor, "B T"]) -> Float[Tensor, "B D L"]:
        scale, shift = self.ss(t)[...,None].chunk(2, dim=1)
        h = self.norm(self.proj_in(x)) * (1+scale) + shift
        return ( x + self.proj_out(h) ) * 2 ** -.5

@dataclass
class DenoiserArgs:
    h_dim: int
    cbam_reduction: int

    c_features: int
    c_dim: int

    a_dim: int
    a_num_stacks: int
    a_stack_depth: int

    scales: list[int]
    stack_depth: int
    block_depth: int
    expand: int

class Denoiser(nn.Module):
    def __init__(
        self,
        args: DenoiserArgs,
    ):
        super().__init__()

        self.audio_features = nn.Sequential(
            nn.Conv1d(A_DIM, args.a_dim, 1),
            WaveNet(args.a_dim, args.a_num_stacks, args.a_stack_depth),
        )

        self.c_map = nn.Sequential(
            RandomFourierFeatures(1 + NUM_LABELS, args.c_features),
            nn.Linear(args.c_features * 2, args.c_dim),
            nn.SiLU(),
        )

        self.proj_h = nn.Conv1d(X_DIM + X_DIM + args.a_dim, args.h_dim, 1)
        self.net = UNet(
            args.h_dim, args.scales,
            VarSequential(*(
                DenoiserUNetBlock(
                    args.h_dim, args.c_dim, args.expand, 
                    CBAM(args.h_dim * args.expand, args.cbam_reduction),
                )
                for _ in range(args.stack_depth)
            )),
            lambda: VarSequential(*(
                DenoiserUNetBlock(
                    args.h_dim, args.c_dim, args.expand,
                    nn.Identity(),
                )
                for _ in range(args.block_depth)
            )),
        )

        self.proj_out = nn.Conv1d(args.h_dim, X_DIM, 1)
        th.nn.init.zeros_(self.proj_out.weight)
        th.nn.init.zeros_(self.proj_out.bias) # type: ignore

    def forward(
        self, 
        audio_features: Float[Tensor, "B A L"],
        positions: Float[Tensor, "B L"],
        label: Float[Tensor, str(f"B {NUM_LABELS}")],
        
        # --- diffusion args --- #
        y: Float[Tensor, str(f"B {X_DIM} L")],  # previous pred_x0
        x: Float[Tensor, str(f"B {X_DIM} L")],  # noised input
        t: Float[Tensor, "B"],                  # (log) denoising step
    ) -> Float[Tensor, str(f"B {X_DIM} L")]:
        c = self.c_map(th.cat([ t[:,None], label ], dim=1))
        h = self.proj_h(th.cat([audio_features,x,y], dim=1))
        return self.proj_out(self.net(h,c))