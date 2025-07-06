
# https://arxiv.org/abs/2312.02696

from typing import Union
from collections.abc import Sequence
from jaxtyping import Float, Int, Shaped

import numpy as np

import torch as th
import torch.nn.functional as F
from torch import nn, Tensor

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
    
def sigmoid(x):
    return F.sigmoid(x) / .542

class Sigmoid(nn.Module):
    def forward(self, x):
        return sigmoid(x)

def add(
    a: Float[Tensor, "..."], 
    b: Float[Tensor, "..."], 
    t: Union[float, Float[Tensor, ""]] = .5,
) -> Float[Tensor, "..."]:
    return ((1-t)*a + t*b) * ((1-t)**2 + t**2) ** -.5

def cat(
    xs: Sequence[Shaped[Tensor, "..."]], 
    dim: int,
) -> Shaped[Tensor, "..."]:
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

class Dropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x: Float[Tensor, "*B"]) -> Float[Tensor, "*B"]:
        if self.training and self.p > 0:
            mask = (th.rand_like(x) > self.p).float()
            return x * mask / (1 - self.p)
        return x


### Magnitude-preserving learned layers

def get_normed_weight(W: Float[Tensor, "O ..."], training: bool, dim=None) -> Float[Tensor, "O ..."]:
    if training and W._version < 2:
        with th.no_grad():
            W.copy_(normalize(W, dim))
    return normalize(W, dim) / np.sqrt(W[0].numel())

class Sum(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.weight_logits = nn.Parameter(th.randn(n))

    def forward(self, xs: list[Float[Tensor, "*B"]]) -> Float[Tensor, "*B"]:
        weight = get_normed_weight(self.weight_logits, self.training).softmax(dim=0)
        weighted_sum = (th.stack(xs, dim=-1) * weight).sum(dim=-1)
        return weighted_sum * weight.pow(2).sum().pow(-.5)

class Linear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, bias=False)
        th.nn.init.normal_(self.weight)

    def forward(self, x: Float[Tensor, "... I"]) -> Float[Tensor, "... O"]:
        return F.linear(x, get_normed_weight(self.weight, self.training))
    
class Embedding(nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        th.nn.init.normal_(self.weight)

    def forward(self, x: Int[Tensor, "... I"]) -> Float[Tensor, "... O"]:
        return F.embedding(
            x,
            get_normed_weight(self.weight, self.training),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )

class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, bias=False, padding_mode='zeros')
        th.nn.init.normal_(self.weight)
    
    def forward(self, x: Float[Tensor, "B iD iL"]) -> Float[Tensor, "B oD oL"]:
        return F.conv1d(
            x,
            get_normed_weight(self.weight, self.training),
            None, self.stride, self.padding, self.dilation, self.groups,
        )

class ConvTranspose1d(nn.ConvTranspose1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, bias=False, padding_mode='zeros')
        th.nn.init.normal_(self.weight)
    
    def forward(self, x: Float[Tensor, "B iD iL"]) -> Float[Tensor, "B oD oL"]:
        assert isinstance(self.padding, tuple)
        output_padding = self._output_padding(
            x,
            None,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            1,
            self.dilation,  # type: ignore[arg-type]
        )
        return F.conv_transpose1d(
            x,
            get_normed_weight(self.weight, self.training, dim=(0,2)),
            None, self.stride, self.padding, output_padding, self.groups, self.dilation,
        )

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