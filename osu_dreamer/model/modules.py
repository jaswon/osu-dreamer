
from functools import partial

import torch
import torch.nn as nn

from einops import rearrange


exists = lambda x: x is not None

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim):
    return nn.ConvTranspose1d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv1d(dim, dim, 4, 2, 1, padding_mode='reflect')

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, max_timescale=10000):
        super().__init__()
        assert dim % 2 == 0
        self.fs = torch.pow(max_timescale, torch.linspace(0, -1, dim // 2))

    def forward(self, t: "N,") -> "N,T":
        # {sin,cos}(t / max_timescale^[0..1])
        embs = t[:, None] * self.fs.to(t.device)[None, :]
        embs = torch.cat([embs.sin(), embs.cos()], dim=-1)
        return embs
    
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        h_dim = dim_head * heads
        self.dim_head = dim_head
        
        self.to_qkv: "N,C,L -> N,3H,L" = nn.Conv1d(dim, h_dim*3, 1, bias=False)
        self.to_out = nn.Conv1d(h_dim, dim, 1)

    def forward(self, x: "N,C,L") -> "N,C,L":
        n, c, l = x.shape
        
        qkv: "3,N,h_dim,L" = self.to_qkv(x).chunk(3, dim=1)
        out = self.attn(*( t.unflatten(1, (self.heads, -1)) for t in qkv ))
        return self.to_out(out)
        
    def attn(self, q: "N,h,d,L", k: "N,h,d,L", v: "N,h,d,L"):
        q = q * self.scale

        sim: "N,h,L,L" = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out:"N,h,L,d" = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h l c -> b (h c) l")
        return out
        
    
class WaveBlock(nn.Module):
    """context is acquired from num_stacks*2**stack_depth neighborhood"""
    
    def __init__(self, dim, stack_depth, num_stacks):
        super().__init__()
        
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    dim, 2 * dim,
                    kernel_size=2,
                    padding=2**i,
                    dilation=2**(i+1),
                    padding_mode='reflect',
                ),
                nn.GLU(dim=1),
            )
            for _ in range(num_stacks)
            for i in range(stack_depth)
        ])
        
    def forward(self, x: "N,C,L") -> "N,C,L":
        h = x
        for net in self.nets:
            h = net(h)
            x = x + h
        return x

class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, emb_dim=None, mult=2, norm=True, groups=1):
        super().__init__()
        
        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, dim),
            )
            if exists(emb_dim)
            else None
        )

        self.ds_conv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim, padding_mode='reflect')

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv1d(dim, dim_out * mult, 3,1,1, padding_mode='reflect', groups=groups),
            nn.SiLU(),

            nn.GroupNorm(1, dim_out * mult),
            nn.Conv1d(dim_out * mult, dim_out, 3,1,1, padding_mode='reflect', groups=groups),
        )

        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: "N,C,L", time_emb: "N,T" = None) -> "N,D,L":
        h: "N,C,L" = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            condition: "N,C" = self.mlp(time_emb)
            h = h + condition.unsqueeze(-1)

        h: "N,D,L" = self.net(h)
        return h + self.res_conv(x)
    
class UNet(nn.Module):
    def __init__(
        self,
        h_dim,
        h_dim_groups,
        dim_mults,
        convnext_mult,
        wave_stack_depth,
        wave_num_stacks,
    ):
        super().__init__()
        
        block = partial(ConvNextBlock, mult=convnext_mult, groups=h_dim_groups)
        
        self.init_conv = nn.Sequential(
            nn.Conv1d(X_DIM+A_DIM+T_DIM, h_dim, 7, padding=3),
            WaveBlock(h_dim, wave_stack_depth, wave_num_stacks),
        )

        dims = [h_dim, *(h_dim*m for m in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_layers = len(in_out)
        
        # time embeddings
        emb_dim = h_dim * 4
        self.time_mlp: "N, -> N,T" = nn.Sequential(
            SinusoidalPositionEmbeddings(h_dim),
            nn.Linear(h_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # layers
        
        self.downs = nn.ModuleList([
            nn.ModuleList([
                block(dim_in, dim_out, emb_dim=emb_dim),
                block(dim_out, dim_out, emb_dim=emb_dim),
                Downsample(dim_out) if ind < (num_layers - 1) else nn.Identity(),
            ])
            for ind, (dim_in, dim_out) in enumerate(in_out)
        ])

        mid_dim = dims[-1]
        self.mid_block1 = block(mid_dim, mid_dim, emb_dim=emb_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block(mid_dim, mid_dim, emb_dim=emb_dim)
        
        self.ups = nn.ModuleList([
            nn.ModuleList([
                block(dim_out * 2, dim_in, emb_dim=emb_dim),
                block(dim_in, dim_in, emb_dim=emb_dim),
                Upsample(dim_in) if ind < (num_layers - 1) else nn.Identity(),
            ])
            for ind, (dim_in, dim_out) in enumerate(in_out[::-1])
        ])

        self.final_conv = nn.Sequential(
            block(h_dim, h_dim),
            zero_module(nn.Conv1d(h_dim, X_DIM, 1)),
        )
        

    def forward(self, x: "N,X,L", a: "N,A,L", t: "N,T,L", ts: "N,") -> "N,X,L":
        
        x: "N,X+A,L" = torch.cat((x,a,t), dim=1)
        
        x: "N,h_dim,L" = self.init_conv(x)

        h = []
        emb: "N,T" = self.time_mlp(ts)

        # downsample
        for block1, block2, downsample in self.downs:
            x: "N,out,L" = block1(x, emb)
            x: "N,out,L" = block2(x, emb)
            h.append(x)
            x: "N,out,L//2" = downsample(x)

        # bottleneck
        x: "N,mid,L" = self.mid_block1(x, emb)
        x: "N,mid,L" = self.mid_attn(x)
        x: "N,mid,L" = self.mid_block2(x, emb)

        # upsample
        for block1, block2, upsample in self.ups:
            x: "N,2*out,L" = torch.cat((x, h.pop()), dim=1)
            x: "N,in,L" = block1(x, emb)
            x: "N,in,L" = block2(x, emb)
            x: "N,in,L*2" = upsample(x)

        return self.final_conv(x)