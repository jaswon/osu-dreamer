
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from osu_dreamer.data.labels import NUM_LABELS

from osu_dreamer.modules.mingru import minGRU2
from osu_dreamer.modules.rff import RandomFourierFeatures


@dataclass
class DenoiserArgs:
    h_dim: int
    depth: int
    expand: int

    rff_dim: int
    c_dim: int

class Denoiser(nn.Module):
    def __init__(
        self,
        dim: int,
        a_dim: int,
        args: DenoiserArgs,
    ):
        super().__init__()

        class emb(nn.Module):
            def __init__(self):
                super().__init__()
                self.rff = RandomFourierFeatures(1, args.rff_dim)
                self.proj_e = nn.Linear(args.rff_dim, args.c_dim, bias=False)
                self.proj_u = nn.Linear(args.rff_dim, 1, bias=False)
                self.proj_label = nn.Linear(NUM_LABELS, args.c_dim, bias=False)

            def forward(
                self,
                t: Float[Tensor, "B"],
                c: Float[Tensor, f"B {NUM_LABELS}"],
            ) -> tuple[Float[Tensor, "B C"], Float[Tensor, "B"]]:
                f = self.rff(t[:,None])
                e = self.proj_e(f) + self.proj_label(c)
                return F.silu(e), self.proj_u(f)[:,0]
        
        self.emb = emb()

        self.proj_h = nn.Conv1d(dim+1, args.h_dim, 1, bias=False)

        class layer(nn.Module):
            def __init__(self):
                super().__init__()
                H = args.h_dim * args.expand
                self.hg = nn.Conv1d(args.h_dim+a_dim, H*2, 1, bias=False)
                self.proj_c = nn.Linear(args.c_dim, H*2, bias=False)
                self.net = nn.Sequential(
                    nn.Conv1d(H, H, 3,1,1, groups=H, bias=False),
                    nn.SiLU(),
                    nn.Conv1d(H, H*2, 1, bias=False),
                    minGRU2(),
                )
                self.out = nn.Conv1d(H, args.h_dim, 1, bias=False)

            def forward(
                self,
                x: Float[Tensor, "B X L"],
                y: Float[Tensor, "B Y L"],
                c: Float[Tensor, "B C"],
            ) -> Float[Tensor, "B X L"]:
                h = th.cat([F.silu(x),y], dim=1)
                c = self.proj_c(c)[:,:,None] + 1
                h,g = (c * self.hg(h)).chunk(2, dim=1)
                h = self.net(h) * F.silu(g)
                return x + self.out(h)
            
        self.layers = nn.ModuleList([ layer() for _ in range(args.depth) ])

        self.proj_out = nn.Conv1d(args.h_dim, dim, 1, bias=False)
        th.nn.init.zeros_(self.proj_out.weight)

    def forward(
        self, 
        a: Float[Tensor, "B A L"],
        label: Float[Tensor, str(f"B {NUM_LABELS}")],
        
        # --- diffusion args --- #
        x: Float[Tensor, "B X L"],  # noised input
        t: Float[Tensor, "B"],      # (log) denoising step
    ) -> tuple[Float[Tensor, "B X L"], Float[Tensor, "B"]]:
        emb, u = self.emb(t, label)
        h = self.proj_h(th.cat([x, th.ones_like(x[:,[0],:])], dim=1))
        for layer in self.layers:
            h = layer(h,a,emb)
        o = self.proj_out(h)
        return o, u