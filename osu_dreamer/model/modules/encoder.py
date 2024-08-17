
from dataclasses import dataclass

from jaxtyping import Float, Int

from torch import nn, Tensor

from osu_dreamer.common.residual import ResStack
from osu_dreamer.common.unet import UNet
from osu_dreamer.common.linear_attn import RoPE, LinearAttn, AttnArgs

@dataclass
class EncoderArgs:
    stack_depth: int
    block_depth: int
    scales: list[int]
    attn_args: AttnArgs

class Encoder(nn.Module):
    def __init__(self, dim: int, args: EncoderArgs, in_dim: int = 0):
        super().__init__()
        self.proj_in = nn.Identity() if in_dim==0 else nn.Conv1d(in_dim, dim, 1)
        
        self.rope = RoPE(args.attn_args.head_dim)
        self.net = UNet(dim, args.scales, ResStack(dim, [
            block
            for _ in range(args.stack_depth)
            for block in [
                LinearAttn(dim, self.rope, args.attn_args),
                nn.Conv1d(dim, dim, 5,1,2, groups=dim),
            ]
        ]), lambda: ResStack(dim, [
            nn.Conv1d(dim, dim, 5,1,2, groups=dim)
            for _ in range(args.block_depth)
        ]))

    def forward(
        self,
        a: Float[Tensor, "B A L"],
        p: Int[Tensor, "B L"],
    ) -> Float[Tensor, "B H L"]:
        h = self.proj_in(a)
        return self.net(h)