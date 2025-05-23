
from jaxtyping import Float, Int

from dataclasses import dataclass

from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.mingru import MinGRU

from .cross_attn import CrossAttn, AttnArgs
from .rope import RoPE

class tokenMixer(nn.Module):
    def __init__(self, dim: int, expand: int):
        super().__init__()
        self.mingru = MinGRU(dim, dim*expand)

    def forward(self, x: Float[Tensor, "B L D"]) -> Float[Tensor, "B L D"]:
        return self.mingru(x.transpose(1,2)).transpose(1,2)
    
class channelMixer(nn.Module):
    def __init__(self, dim: int, expand: int):
        super().__init__()
        self.proj_h = MP.Linear(dim, dim*expand)
        self.proj_g = MP.Linear(dim, dim*expand)
        self.proj_out = MP.Linear(dim * expand, dim)

    def forward(
        self,
        x: Float[Tensor, "B L D"],
    ) -> Float[Tensor, "B L D"]:
        h,g = self.proj_h(x), MP.silu(self.proj_g(x))
        return self.proj_out(h*g)
    
class condBlock(nn.Module):
    def __init__(self, dim: int, c_dim: int, op: nn.Module):
        super().__init__()
        self.op = op
        self.scale = nn.Sequential( MP.Linear(c_dim, dim), MP.Gain() )
        self.shift = nn.Sequential( MP.Linear(c_dim, dim), MP.Gain() )
        self.alpha = nn.Sequential( MP.Linear(c_dim, dim), MP.Gain() )

    def forward(
        self,
        x: Float[Tensor, "B L D"],
        c: Float[Tensor, "B C"],
        *args,
    ) -> Float[Tensor, "B L D"]:
        r = MP.normalize(x, dim=-1)
        r = r * (1+self.scale(c)[:,None,:]) + self.shift(c)[:,None,:]
        r = self.op(r, *args)
        r = self.alpha(c)[:,None,:] * r
        return x + r
    
class DecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ctx_dim: int,
        c_dim: int,
        expand: int,
        rope: RoPE,
        attn_args: AttnArgs,
    ):
        super().__init__()
        self.seq_mixer = condBlock(dim, c_dim, tokenMixer(dim, expand))
        self.ctx_mixer = condBlock(dim, c_dim, CrossAttn(dim, ctx_dim, rope, attn_args))
        self.chn_mixer = condBlock(dim, c_dim, channelMixer(dim, expand))

    def forward(
        self,
        x: Float[Tensor, "B N E"],
        x_t: Int[Tensor, "B N"],
        ctx: Float[Tensor, "B L H"],
        ctx_t: Int[Tensor, "B L"],
        c: Float[Tensor, "B C"],
    ) -> Float[Tensor, "B N E"]:
        x = self.seq_mixer(x,c)
        x = self.ctx_mixer(x,c,x_t,ctx,ctx_t)
        x = self.chn_mixer(x,c)
        return x

@dataclass
class DecoderArgs:
    depth: int
    expand: int
    attn_args: AttnArgs

class Decoder(nn.Module):
    def __init__(
        self,
        dim: int,
        ctx_dim: int,
        c_dim: int,
        args: DecoderArgs,
        seq_len: int,
    ):
        super().__init__()
        self.rope = RoPE(args.attn_args.head_dim, 2*seq_len)
        self.blocks = nn.ModuleList([
            DecoderBlock(dim, ctx_dim, c_dim, args.expand, self.rope, args.attn_args)
            for _ in range(args.depth)
        ])

    def forward(
        self,
        x: Float[Tensor, "B N E"],
        x_t: Int[Tensor, "B N"],
        ctx: Float[Tensor, "B L H"],
        ctx_t: Int[Tensor, "B L"],
        c: Float[Tensor, "B C"],
    ) -> Float[Tensor, "B N E"]:
        for block in self.blocks:
            x = checkpoint(block, x, x_t, ctx, ctx_t, c, use_reentrant=False) # type: ignore
        return x