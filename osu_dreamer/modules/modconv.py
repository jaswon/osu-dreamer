
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from einops import rearrange

class ModulatedConv1d(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        t_dim: int,
        kernel_size: int,
        *args, **kwargs,
    ):
        super().__init__()
        if kernel_size == 1:
            self.conv = nn.Identity()
        else:
            self.conv = nn.Conv1d(in_dim, in_dim, kernel_size, *args, **kwargs, groups=in_dim, bias=False)
        self.proj = nn.Conv1d(in_dim, out_dim, 1)
        self.mod = nn.Linear(t_dim, in_dim)

    def forward(self, xt: tuple[
        Float[Tensor, "B I L"], 
        Float[Tensor, "B T"],
    ]) -> Float[Tensor, "B O L"]:
        bx,t = xt
        b = t.size(0)
        mod = self.mod(t)[:,None,:,None] # b 1 i 1
        bw = self.proj.weight * (mod+1) # b o i k
        bw = bw * th.rsqrt(th.sum(bw ** 2, dim=[2,3]) + 1e-8)[:,:,None,None]

        w = rearrange(bw, 'b o i k -> (b o) i k')
        x = rearrange(self.conv(bx), 'b d l -> 1 (b d) l')
        o = F.conv1d(x, w, groups=b)
        bo = rearrange(o, '1 (b d) l -> b d l', b=b)
        bo = bo + self.proj.bias[None,:,None] # type: ignore
        return bo