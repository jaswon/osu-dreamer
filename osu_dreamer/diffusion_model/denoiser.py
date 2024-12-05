
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.data.labels import NUM_LABELS

import osu_dreamer.modules.mp as MP
from .modules import Seq


@dataclass
class DenoiserArgs:
    h_dim: int
    c_dim: int
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

        self.emb_f = MP.Linear(f_dim, args.c_dim)
        self.emb_l = MP.Linear(NUM_LABELS, args.c_dim)

        self.proj_h = MP.Conv1d(dim+1, args.h_dim, 1)

        class layer(nn.Module):
            def __init__(self):
                super().__init__()
                H = args.h_dim * args.expand
                self.proj_c = nn.Sequential(
                    MP.Linear(args.c_dim, args.h_dim+a_dim),
                    MP.Gain(),
                )
                self.proj_h = nn.Sequential(
                    MP.SiLU(),
                    MP.Conv1d(args.h_dim+a_dim, args.h_dim, 1),
                )
                self.seq = Seq(args.h_dim, H)

            def forward(
                self,
                x: Float[Tensor, "B X L"],
                y: Float[Tensor, "B Y L"],
                c: Float[Tensor, "B C"],
            ) -> Float[Tensor, "B X L"]:
                x = MP.pixel_norm(x)
                c = self.proj_c(c)[:,:,None] + 1
                h = self.proj_h(c * MP.cat([x,y], dim=1))
                return MP.add(x, self.seq(h), t=.3)
            
        self.layers = nn.ModuleList([ layer() for _ in range(args.depth) ])

        self.proj_out = nn.Sequential(
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
        emb = MP.silu(MP.add(self.emb_f(f), self.emb_l(label-5)))
        h = self.proj_h(MP.cat([x, th.ones_like(x[:,:1,:])], dim=1))
        for layer in self.layers:
            h = layer(h,a,emb)
        o = self.proj_out(h)
        return o