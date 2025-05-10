
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.data.labels import NUM_LABELS

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.mingru import MinGRU2
from osu_dreamer.modules.attend_label import AttendLabel


@dataclass
class DenoiserArgs:
    h_dim: int
    depth: int
    label_head_dim: int
    label_num_heads: int

class sequenceMixer(nn.Module):
    def __init__(self, dim: int, a_dim: int, f_dim: int, args: DenoiserArgs):
        super().__init__()
        self.net = nn.Sequential(
            MP.Conv1d(args.h_dim, args.h_dim, 3,1,1, groups=args.h_dim),
            MP.Conv1d(args.h_dim, 2*args.h_dim, 1),
            MinGRU2(),
        )

    def forward(
        self,
        x: Float[Tensor, "B X L"],
        c: Float[Tensor, "B C"],
    ) -> Float[Tensor, "B X L"]:
        return self.net(x)
    
class labelMixer(nn.Module):
    def __init__(self, dim: int, a_dim: int, f_dim: int, args: DenoiserArgs):
        super().__init__()
        self.net = AttendLabel(
            args.h_dim, f_dim + NUM_LABELS,
            head_dim = args.label_head_dim,
            num_heads = args.label_num_heads,
        )

    def forward(
        self,
        x: Float[Tensor, "B X L"],
        c: Float[Tensor, "B C"],
    ) -> Float[Tensor, "B X L"]:
        return self.net(x,c)
    
class channelMixer(nn.Module):
    def __init__(self, dim: int, a_dim: int, f_dim: int, args: DenoiserArgs):
        super().__init__()
        self.proj_in = MP.Conv1d(args.h_dim, args.h_dim, 1)
        self.proj_h = nn.Sequential(
            MP.Conv1d(args.h_dim, args.h_dim, 3,1,1, groups=args.h_dim),
            MP.Conv1d(args.h_dim, args.h_dim, 1),
        )
        self.proj_g = nn.Sequential(
            MP.Conv1d(args.h_dim, args.h_dim, 3,1,1, groups=args.h_dim),
            MP.Conv1d(args.h_dim, args.h_dim, 1),
            MP.SiLU(),
        )
        self.proj_out = MP.Conv1d(args.h_dim, args.h_dim, 1)

    def forward(
        self,
        x: Float[Tensor, "B X L"],
        c: Float[Tensor, "B C"],
    ) -> Float[Tensor, "B X L"]:
        x = self.proj_in(x)
        h,g = self.proj_h(x), self.proj_g(x)
        return self.proj_out(h*g)

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
            
        self.net = MP.ResNet([ 
            layer(dim, a_dim, f_dim, args)
            for _ in range(args.depth)
            for layer in [
                sequenceMixer,
                labelMixer,
                channelMixer,
            ]
        ])

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
        c = th.cat([f, label], dim=1)
        h = self.proj_h(MP.cat([a,x], dim=1))
        h = self.net(h,c)
        return self.proj_out(h)