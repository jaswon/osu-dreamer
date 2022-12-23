
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

class LinearAttention(Attention):
    """https://arxiv.org/abs/1812.01243"""
    
    def attn(self, q: "N,h,d,L", k: "N,h,d,L", v: "N,h,d,L"):
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)

        ctx: "N,h,d,d" = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out: "N,h,d,l" = torch.einsum("b h d e, b h d n -> b h e n", ctx, q)
        out = rearrange(out, "b h c l -> b (h c) l")
        return out
    
class WaveBlock(nn.Module):
    """context is acquired from num_stacks*2**stack_depth neighborhood"""
    
    def __init__(self, dim, stack_depth, num_stacks, mult=1, h_dim_groups=1, up=False):
        super().__init__()

        self.in_net = nn.Conv1d(dim, dim * mult, 1)
        self.out_net = nn.Conv1d(dim * mult, dim, 1)
        
        self.nets = nn.ModuleList([
            nn.Sequential(
                (nn.ConvTranspose1d if up else nn.Conv1d)(
                    in_channels=dim * mult, 
                    out_channels=2 * dim * mult,
                    kernel_size=2,
                    padding=2**i,
                    dilation=2**(i+1),
                    groups=h_dim_groups,
                    **({} if up else dict(padding_mode='replicate')),
                ),
                nn.GLU(dim=1),
            )
            for _ in range(num_stacks)
            for i in range(stack_depth)
        ])
        
    def forward(self, x: "N,C,L") -> "N,C,L":
        x = self.in_net(x)
        h = x
        for net in self.nets:
            h = net(h)
            x = x + h
        return self.out_net(x)

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
            nn.Conv1d(dim, dim_out * mult, 7,1,3, padding_mode='reflect', groups=groups),
            nn.SiLU(),

            nn.GroupNorm(1, dim_out * mult),
            nn.Conv1d(dim_out * mult, dim_out, 7,1,3, padding_mode='reflect', groups=groups),
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
        in_dim,
        out_dim,
        h_dims,
        h_dim_groups,
        convnext_mult,
        wave_stack_depth,
        wave_num_stacks,
        blocks_per_depth,
        attn_heads,
        attn_dim,
    ):
        super().__init__()
        
        block = partial(ConvNextBlock, mult=convnext_mult, groups=h_dim_groups)

        in_out = list(zip(h_dims[:-1], h_dims[1:]))
        num_layers = len(in_out)
        
        self.init_conv = nn.Sequential(
            nn.Conv1d(in_dim, h_dims[0], 7, padding=3),
            WaveBlock(h_dims[0], wave_stack_depth, wave_num_stacks),
        )
        
        # time embeddings
        emb_dim = h_dims[0] * 4
        self.time_mlp: "N, -> N,T" = nn.Sequential(
            SinusoidalPositionEmbeddings(h_dims[0]),
            nn.Linear(h_dims[0], emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # layers
        
        self.downs = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    block(dim_in if i==0 else dim_out, dim_out, emb_dim=emb_dim)
                    for i in range(blocks_per_depth)
                ]),
                nn.ModuleList([
                    Residual(PreNorm(dim_out, LinearAttention(dim_out, heads=attn_heads, dim_head=attn_dim)))
                    for _ in range(blocks_per_depth)
                ]),
                Downsample(dim_out) if ind < (num_layers - 1) else nn.Identity(),
            ])
            for ind, (dim_in, dim_out) in enumerate(in_out)
        ])

        mid_dim = h_dims[-1]
        self.mid_block1 = block(mid_dim, mid_dim, emb_dim=emb_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, heads=attn_heads, dim_head=attn_dim)))
        self.mid_block2 = block(mid_dim, mid_dim, emb_dim=emb_dim)
        
        self.ups = nn.ModuleList([
            nn.ModuleList([
                nn.ModuleList([
                    block(dim_out * 2 if i==0 else dim_in, dim_in, emb_dim=emb_dim)
                    for i in range(blocks_per_depth)
                ]),
                nn.ModuleList([
                    Residual(PreNorm(dim_in, LinearAttention(dim_in, heads=attn_heads, dim_head=attn_dim)))
                    for _ in range(blocks_per_depth)
                ]),
                Upsample(dim_in) if ind < (num_layers - 1) else nn.Identity(),
            ])
            for ind, (dim_in, dim_out) in enumerate(in_out[::-1])
        ])

        self.final_conv = nn.Sequential(
            *(
                block(h_dims[0], h_dims[0])
                for _ in range(blocks_per_depth)
            ),
            zero_module(nn.Conv1d(h_dims[0], out_dim, 1)),
        )
        

    def forward(self, x: "N,X,L", a: "N,A,L", ts: "N,") -> "N,X,L":
        
        x: "N,X+A,L" = torch.cat([x,a], dim=1)
        
        x: "N,h_dim,L" = self.init_conv(x)

        h = []
        emb: "N,T" = self.time_mlp(ts)

        # downsample
        for blocks, attns, downsample in self.downs:
            for block, attn in zip(blocks, attns):
                x = attn(block(x, emb))
            h.append(x)
            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, emb)

        # upsample
        for blocks, attns, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            for block, attn in zip(blocks, attns):
                x = attn(block(x, emb))
            x = upsample(x)

        return self.final_conv(x)