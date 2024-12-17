
# https://arxiv.org/abs/2312.02696

from typing import Union
from collections.abc import Sequence
from jaxtyping import Complex, Float

import numpy as np

import torch as th
import torch.nn.functional as F
from torch import nn, Tensor

def normalize(x: Float[Tensor, "..."], dim=None, eps=1e-4) -> Float[Tensor, "..."]:
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = th.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=th.float32)
    norm = th.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)

def pixel_norm(x: Float[Tensor, "B D ..."]) -> Float[Tensor, "B D ..."]:
    return normalize(x, dim=1)

class PixelNorm(nn.Module):
    def forward(self, x):
        return pixel_norm(x)

def M(a):
    return th.linalg.vector_norm(a, dim=1).flatten() * a.size(1) ** -.5

### Magnitude-preserving fixed function layers

def silu(x):
    return F.silu(x) / .596

class SiLU(nn.Module):
    def forward(self, x):
        return silu(x)

def add(
    a: Float[Tensor, "..."], 
    b: Float[Tensor, "..."], 
    t: Union[float, Float[Tensor, ""]] = .5,
) -> Float[Tensor, "..."]:
    return ((1-t)*a + t*b) * ((1-t)**2 + t**2) ** -.5

def cat(
    xs: Sequence[Float[Tensor, "..."]], 
    dim: int,
) -> Float[Tensor, "..."]:
    Ns = [ x.size(dim) for x in xs ]
    return np.mean(Ns) ** .5 * th.cat([
        x * n ** -.5
        for n,x in zip(Ns,xs)
    ], dim=dim)

class Gain(nn.Module):
    def __init__(self):
        super().__init__()
        self.g = nn.Parameter(th.tensor(0.))

    def forward(self, x: Float[Tensor, "..."]) -> Float[Tensor, "..."]:
        return x * self.g


### Magnitude-preserving learned layers

def get_normed_weight(W: Float[Tensor, "O ..."], training: bool) -> Float[Tensor, "O ..."]:
    if training:
        with th.no_grad():
            W.copy_(normalize(W))
    return normalize(W) / np.sqrt(W[0].numel())

class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, bias=False)
        th.nn.init.normal_(self.weight)

    def forward(self, x: Float[Tensor, "... I"]) -> Float[Tensor, "... O"]:
        return F.linear(x, get_normed_weight(self.weight, self.training))

class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, bias=False)
        th.nn.init.normal_(self.weight)
    
    def forward(self, x: Float[Tensor, "B iD iL"]) -> Float[Tensor, "B oD oL"]:
        return self._conv_forward(x, get_normed_weight(self.weight, self.training), None)

class Conv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, bias=False)
        th.nn.init.normal_(self.weight)
    
    def forward(self, x: Float[Tensor, "B iD iH iW"]) -> Float[Tensor, "B oD oH oW"]:
        return self._conv_forward(x, get_normed_weight(self.weight, self.training), None)


### Magnitude-preserving misc. layers
    
class RandomFourierFeatures(nn.Module):
    def __init__(self, dim: int, n_feats: int):
        super().__init__()
        self.register_buffer('f', 2 * th.pi * th.randn(dim, n_feats))
        self.register_buffer('p', 2 * th.pi * th.rand(n_feats))

    def forward(self, x: Float[Tensor, "B C"]) -> Float[Tensor, "B N"]:
        return 2**.5 * th.cos(x @ self.f + self.p)
    
def complex_log(float_input: Float[Tensor, "..."], eps=1e-6) -> Complex[Tensor, "..."]:
    real = th.clamp_min(float_input.abs(), eps).log()
    imag = (float_input < 0) * th.pi
    return th.complex(real, imag.float())

@th.compiler.disable()
def min_gru(
    h: Float[Tensor, "... L"], 
    g: Float[Tensor, "... L"],
) -> Float[Tensor, "... L"]:
    log_scale = .5 * th.log(1/th.cosh(g)+1) # mp
    log_coeffs = -F.softplus(g) + log_scale
    log_values = -F.softplus(-g) + log_scale + complex_log(h)

    # heinsen associative scan (log-space)
    a_star = log_coeffs.cumsum(dim=-1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=-1)
    log_h = a_star + log_h0_plus_b_star
    
    return log_h.exp().real

class minGRU2(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        assert dim%2==0
        self.fb_hg = Conv1d(dim, dim*2, 1)

    def forward(
        self,
        x: Float[Tensor, "B H L"],
    ) -> Float[Tensor, "B H L"]:
        fore_hg, back_hg = self.fb_hg(x).chunk(2, dim=1)
        return cat([
            min_gru(*fore_hg.chunk(2, dim=1)),
            min_gru(*back_hg.flip(2).chunk(2, dim=1)).flip(2),
        ], dim=1)
    
class Seq(nn.Module):
    def __init__(self, dim: int, h_dim: int = -1):
        super().__init__()
        h_dim = dim if h_dim==-1 else h_dim
        self.h = nn.Sequential(
            SiLU(),
            Conv1d(dim, h_dim, 1),
            Conv1d(h_dim, h_dim, 3,1,1, groups=h_dim),
            SiLU(),
            minGRU2(h_dim),
            PixelNorm(),
        )
        self.g = nn.Sequential(
            SiLU(),
            Conv1d(dim, h_dim, 1),
            SiLU(),
        )
        self.out = Conv1d(h_dim, dim, 1)

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L"]:
        return self.out(self.h(x) * self.g(x))
    
class ResNet(nn.Module):
    def __init__(self, nets: Sequence[nn.Module]):
        super().__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, x: Float[Tensor, "B D L"], *args, **kwargs) -> Float[Tensor, "B D L"]:
        for net in self.nets:
            x = pixel_norm(x)
            x = add(x, net(x,*args,**kwargs), t=.1)
        return x