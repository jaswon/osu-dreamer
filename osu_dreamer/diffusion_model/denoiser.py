
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
        
        self.emb = nn.Sequential(
            MP.Linear(f_dim+NUM_LABELS, args.c_dim),
            MP.SiLU(),
        )

        self.proj_h = MP.Conv1d(dim+1, args.h_dim, 1)

        class layer(nn.Module):
            def __init__(self):
                super().__init__()
                H = args.h_dim * args.expand
                self.hg = MP.Conv1d(args.h_dim+a_dim, H*2, 1)
                self.proj_c = nn.Sequential(
                    MP.Linear(args.c_dim, H*2),
                    MP.Gain(),
                )
                self.net = nn.Sequential(
                    MP.Conv1d(H, H, 3,1,1, groups=H),
                    MP.SiLU(),
                    MP.minGRU2(H),
                )
                self.out = MP.Conv1d(H, args.h_dim, 1)

            def forward(
                self,
                x: Float[Tensor, "B X L"],
                y: Float[Tensor, "B Y L"],
                c: Float[Tensor, "B C"],
            ) -> Float[Tensor, "B X L"]:
                x = MP.pixel_norm(x)
                h = MP.cat([MP.silu(x),y], dim=1)
                c = self.proj_c(c)[:,:,None] + 1
                h,g = (c * self.hg(h)).chunk(2, dim=1)
                h = self.net(h) * MP.silu(g)
                return MP.add(x, self.out(h), t=.3)
            
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
        emb = self.emb(MP.cat([f, label], dim=1))
        h = self.proj_h(MP.cat([x, th.ones_like(x[:,:1,:])], dim=1))
        for layer in self.layers:
            h = layer(h,a,emb)
        o = self.proj_out(h)
        return o