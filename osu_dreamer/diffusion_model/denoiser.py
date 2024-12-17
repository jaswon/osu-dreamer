
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.data.labels import NUM_LABELS

import osu_dreamer.modules.mp as MP


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
        self.emb_l = nn.Sequential(
            MP.Linear(NUM_LABELS, args.c_dim),
            MP.PixelNorm(),
        )

        self.proj_h = MP.Conv1d(dim+1, args.h_dim, 1)

        H = args.h_dim * args.expand
        class layer(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_y = MP.Conv1d(a_dim, args.h_dim, 1)
                self.proj_c = nn.Sequential(
                    MP.Linear(args.c_dim, args.h_dim),
                    MP.Gain(),
                )
                self.seq = MP.Seq(args.h_dim, H)

            def forward(
                self,
                x: Float[Tensor, "B X L"],
                y: Float[Tensor, "B Y L"],
                c: Float[Tensor, "B C"],
            ) -> Float[Tensor, "B X L"]:
                xy = MP.add(x, self.proj_y(y), t=.1)
                c = self.proj_c(c)[:,:,None] + 1
                return self.seq(c * xy)
            
        self.net = MP.ResNet([ layer() for _ in range(args.depth) ])

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
        h = self.net(h,MP.pixel_norm(a),emb)
        return self.proj_out(h)