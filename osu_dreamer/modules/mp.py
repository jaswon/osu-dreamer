
# https://arxiv.org/abs/2312.02696

from typing import Union
from collections.abc import Sequence
from jaxtyping import Float

import numpy as np

import torch as th
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.checkpoint import checkpoint

def normalize(x: Float[Tensor, "B ..."], dim=None, eps=1e-4) -> Float[Tensor, "B ..."]:
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