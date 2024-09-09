
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.data.beatmap.encode import X_DIM
from osu_dreamer.data.load_audio import A_DIM
from osu_dreamer.data.prepare_map import NUM_LABELS

from .modules.scaleshift import ScaleShift
from .modules.residual import ResStack
from .modules.unet import UNet
from .modules.cbam import CBAM
from .modules.rff import RandomFourierFeatures

@dataclass
class DenoiserArgs:
    h_dim: int

    c_features: int
    c_dim: int

    a_features: int
    a_num_stacks: int
    a_stack_depth: int

    scales: list[int]
    block_depth: int
    stack_depth: int

class Denoiser(nn.Module):
    def __init__(
        self,
        args: DenoiserArgs,
    ):
        super().__init__()

        self.proj_a = nn.Conv1d(A_DIM, args.a_features, 1)
        self.a_map = ResStack(args.a_features, [
            nn.Sequential(
                nn.ZeroPad1d((1,0) if d==0 else 2**(d-1)),
                nn.Conv1d(args.a_features, args.a_features, 2, dilation=2**d),
            )
            for _ in range(args.a_num_stacks)
            for d in range(args.a_stack_depth)
        ]) # receptive field = 1+s*(2**d-1)

        self.c_map = nn.Sequential(
            RandomFourierFeatures(2 + NUM_LABELS, args.c_features),
            nn.Linear(args.c_features * 2, args.c_dim),
            nn.SiLU(),
        )

        self.proj_h = nn.Conv1d(X_DIM + X_DIM + args.a_features, args.h_dim, 1)
        
        self.net = UNet(
            args.h_dim, args.scales,
            ResStack(args.h_dim, [
                ScaleShift(args.h_dim, args.c_dim, block)
                for _ in range(args.stack_depth)
                for block in [
                    nn.Conv1d(args.h_dim, args.h_dim, 5,1,2, groups=args.h_dim),
                    CBAM(args.h_dim),
                ]
            ]),
            lambda: ResStack(args.h_dim, [
                ScaleShift(args.h_dim, args.c_dim, nn.Conv1d(args.h_dim, args.h_dim, 5,1,2, groups=args.h_dim))
                for _ in range(args.block_depth)
            ]),
        )

        self.proj_out = nn.Conv1d(args.h_dim, X_DIM, 1)
        th.nn.init.zeros_(self.proj_out.weight)
        th.nn.init.zeros_(self.proj_out.bias) # type: ignore

    def encode_audio(self, audio: Float[Tensor, str(f"B {A_DIM} L")]) -> Float[Tensor, "B A L"]:
        return self.a_map(self.proj_a(audio))

    def forward(
        self, 
        audio: Float[Tensor, "B A L"],
        star_rating: Float[Tensor, "B 1"],
        diff_labels: Float[Tensor, str(f"B {NUM_LABELS}")],
        
        # --- diffusion args --- #
        y: Float[Tensor, str(f"B {X_DIM} L")],  # previous pred_x0
        x: Float[Tensor, str(f"B {X_DIM} L")],  # noised input
        t: Float[Tensor, "B"],                  # (log) denoising step
    ) -> Float[Tensor, str(f"B {X_DIM} L")]:
        c = self.c_map(th.cat([ t[:,None], star_rating, diff_labels ], dim=1))
        h = self.proj_h(th.cat([audio,x,y], dim=1))
        return self.proj_out(self.net(h,c))