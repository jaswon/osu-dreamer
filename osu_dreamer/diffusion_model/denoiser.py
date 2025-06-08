
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.data.labels import NUM_LABELS

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.dit import DiT, DiTArgs


class FourierFeatures(nn.Module):
    def __init__(self, dim: int, features: int):
        super().__init__()
        self.proj_in = nn.Linear(dim, features)
        th.nn.init.normal_(self.proj_in.weight)
        th.nn.init.uniform_(self.proj_in.bias, -th.pi, th.pi)
        self.scale = (2/features) ** .5

    def forward(self, x: Float[Tensor, "*B I"]) -> Float[Tensor, "*B O"]:
        return self.scale * th.cos(self.proj_in(x))

@dataclass
class DenoiserArgs:
    h_dim: int
    f_dim: int
    noise_level_features: int
    depth: int
    expand: int

class Denoiser(nn.Module):
    def __init__(
        self,
        dim: int,
        a_dim: int,
        args: DenoiserArgs,
    ):
        super().__init__()

        self.nlf = nn.Sequential(
            FourierFeatures(1, args.noise_level_features),
            MP.Linear(args.noise_level_features, args.f_dim),
        )
        self.proj_label = nn.Sequential(
            MP.Linear(NUM_LABELS, args.f_dim),
            MP.SiLU(),
            MP.Linear(args.f_dim, args.f_dim),
        )
        self.proj_c = nn.Sequential(
            nn.SiLU(),
            nn.Linear(args.f_dim, args.f_dim),
            nn.SiLU(),
        )
            
        self.proj_h = MP.Conv1d(dim+a_dim, args.h_dim, 1)
        self.net = DiT(args.h_dim, args.f_dim, DiTArgs(args.depth, args.expand))
        self.proj_out = nn.Sequential(
            MP.PixelNorm(),
            MP.Conv1d(args.h_dim, dim, 1),
            MP.Gain(),
        )

    def forward(
        self, 
        a: Float[Tensor, "B A L"],
        label: Float[Tensor, str(f"B {NUM_LABELS}")],
        
        # --- diffusion args --- #
        x: Float[Tensor, "B X L"],  # noised input
        t: Float[Tensor, "B"],      # noise level
    ) -> Float[Tensor, "B X L"]:
        c = MP.silu(self.nlf(t[:,None]) + self.proj_label(label))
        h = self.proj_h(MP.cat([a,x], dim=1))
        h = self.net(h,c)
        return self.proj_out(h)