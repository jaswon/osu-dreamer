
from dataclasses import dataclass

from jaxtyping import Float

from torch import nn, Tensor

from osu_dreamer.data.labels import NUM_LABELS

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.dit import DiT


@dataclass
class DenoiserArgs:
    h_dim: int
    depth: int
    expand: int

class Denoiser(nn.Module):
    def __init__(
        self,
        dim: int,
        a_dim: int,
        f_dim: int,
        args: DenoiserArgs,
    ):
        super().__init__()

        self.proj_h = MP.Conv1d(dim+a_dim, args.h_dim, 1)
        self.proj_label = nn.Sequential(
            MP.Linear(NUM_LABELS, f_dim),
            MP.SiLU(),
            MP.Linear(f_dim, f_dim),
        )
        self.proj_f = nn.Sequential(
            MP.Linear(f_dim, f_dim),
            MP.SiLU(),
            MP.Linear(f_dim, f_dim),
        )
            
        self.net = DiT(args.h_dim, f_dim, args.depth, args.expand)

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
        f: Float[Tensor, "B F"],    # noise level features
    ) -> Float[Tensor, "B X L"]:
        c = MP.silu(self.proj_f(f) + self.proj_label(label))
        h = self.proj_h(MP.cat([a,x], dim=1))
        h = self.net(h,c)
        return self.proj_out(h)