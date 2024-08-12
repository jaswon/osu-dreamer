
from dataclasses import dataclass

from typing import Optional
from jaxtyping import Float, Int

from torch import nn, Tensor
import torch as th
import torch.nn.functional as F

from osu_dreamer.common.residual import ResStack
from osu_dreamer.common.unet import UNet
from osu_dreamer.common.linear_attn import RoPE, LinearAttn, AttnArgs

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(th.ones(dim,1))

    def forward(self, x):
        return F.normalize(x, dim = 1, eps=1e-8) * self.gamma * self.scale

class ScaleShift(nn.Module):
    def __init__(self, dim: int, cond_dim: int, net: nn.Module):
        super().__init__()
        self.net = net

        self.norm = RMSNorm(dim)
        self.to_scale_shift = nn.Linear(cond_dim, dim * 2)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: Float[Tensor, "B D L"], e: Float[Tensor, "B T"]):
        scale, shift = self.to_scale_shift(e).unsqueeze(-1).chunk(2, dim=1)
        return self.net(self.norm(x) * (1+scale) + shift)

@dataclass
class EncoderArgs:
    stack_depth: int
    scales: list[int]
    attn_args: AttnArgs

class Encoder(nn.Module):
    def __init__(self, dim: int, args: EncoderArgs, in_dim: int = 0, t_dim: int = 0):
        super().__init__()
        self.rope = RoPE(args.attn_args.head_dim)
        self.proj_in = nn.Identity() if in_dim==0 else nn.Conv1d(in_dim, dim, 1)

        self.use_cond = t_dim != 0

        self.net = UNet(dim, args.scales, ResStack(dim, [
            ScaleShift(dim, t_dim, block) if self.use_cond else block
            for _ in range(args.stack_depth)
            for block in [
                LinearAttn(dim, self.rope, args.attn_args),
                nn.Conv1d(dim, dim, 5,1,2, groups=dim),
            ]
        ]))

        self.chunk_size = 1
        for scale in args.scales:
            self.chunk_size *= scale

    def forward(
        self,
        a: Float[Tensor, "B A L"],
        p: Int[Tensor, "B L"],
        t: Optional[Float[Tensor, "B T"]] = None,
    ) -> Float[Tensor, "B H L"]:
        self.rope.set_ts( p.float()[...,::self.chunk_size] / self.chunk_size )
        h = self.proj_in(a)
        if self.use_cond:
            return self.net(h, t)
        else:
            return self.net(h)