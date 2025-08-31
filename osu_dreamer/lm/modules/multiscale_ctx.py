
from jaxtyping import Float, Int

import torch as th
from torch import nn, Tensor

class MultiScaleEncoder(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, r: list[tuple[int, int]]):
        super().__init__()
        self.r = r
        self.out = nn.Linear(in_dim, out_dim)

        last_w = 1
        self.convs = nn.ModuleList([ nn.Identity() ])
        for r_past, r_future in r[:-1]:
            w = 1+r_past+r_future
            self.convs.append(nn.Sequential(
                nn.ZeroPad1d((r_past*last_w, r_future*last_w)),
                nn.Conv1d(in_dim, in_dim, w, 1, dilation=last_w, groups=in_dim),
                nn.Conv1d(in_dim, in_dim, 1),
            ))
            last_w *= w

    def forward(
        self, 
        x: Float[Tensor, "1 D L"], 
        positions: Int[Tensor, "B N"],
    ) -> Float[Tensor, "B N T D"]:
        
        ctx = [ x.transpose(1,2)[0, positions[:,:,None]] ] # B N 1 D
        stride = 1
        for (r_past, r_future), conv in zip(self.r, self.convs):
            x = conv(x)
            w = th.cat([ -1-th.arange(r_past), 1+th.arange(r_future) ], dim=0)
            p = positions[:,:,None] + stride * w[None,None,:] # B N W
            ctx.append(x.transpose(1,2)[0, p]) # B N W D
            stride *= 1+r_past+r_future

        return self.out(th.cat(ctx, dim=2))
