
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from osu_dreamer.data.prepare_map import NUM_LABELS

from osu_dreamer.modules.mingru import minGRU2
from osu_dreamer.modules.modconv import ModulateConv
from osu_dreamer.modules.wavenet import ResSkipNet


@dataclass
class DenoiserArgs:
    h_dim: int
    depth: int
    expand: int

    c_dim: int
    c_depth: int

    mod_depth: int

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

        mod = ModulateConv(args.c_dim, args.mod_depth)
        self.mod = mod

        self.proj_h = mod(nn.Conv1d(dim, args.h_dim, 1))

        class layer(nn.Module):
            def __init__(self):
                super().__init__()
                H = args.h_dim * args.expand
                self.hg = nn.Sequential(
                    mod(nn.Conv1d(args.h_dim+a_dim, H*2, 1)),
                    nn.SiLU(),
                    mod(nn.Conv1d(H*2, H*2, 1)),
                )
                self.net = nn.Sequential(
                    mod(nn.Conv1d(H, H, 3,1,1, groups=H)),
                    nn.SiLU(),
                    mod(nn.Conv1d(H, H*2, 1)),
                    minGRU2(),
                )
                self.out = mod(nn.Conv1d(H, args.h_dim*2, 1))
        
            def forward(
                self,
                x: Float[Tensor, "B X L"],
                y: Float[Tensor, "B Y L"],
            ) -> Float[Tensor, "B X*2 L"]:
                h,g = self.hg(th.cat([x,y], dim=1)).chunk(2, dim=1)
                h = self.net(h) * F.silu(g)
                return self.out(h)
            
        self.net = ResSkipNet(args.h_dim, [ layer() for _ in range(args.depth) ])

        self.proj_out = nn.Conv1d(args.h_dim, dim, 1)
        th.nn.init.zeros_(self.proj_out.weight)
        th.nn.init.zeros_(self.proj_out.bias) # type: ignore

    def forward(
        self, 
        a: Float[Tensor, "B A L"],
        label: Float[Tensor, str(f"B {NUM_LABELS}")],
        
        # --- diffusion args --- #
        x: Float[Tensor, "B X L"],  # noised input
        t: Float[Tensor, "B"],      # (log) denoising step
    ) -> Float[Tensor, "B X L"]:
        c = self.proj_c(th.cat([t[:,None],label], dim=1))
        with self.mod.set(c):
            h = self.proj_h(x)
            h = self.net(h,a)
        return self.proj_out(h)