
from dataclasses import dataclass
from typing import Sequence
from jaxtyping import Float, Complex

import numpy as np
import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from einops import repeat

from pykeops.torch import Genred


def broadcast_dims(*tensors: Tensor) -> Sequence[Tensor]:
    """pads the shape of all tensors to match the largest ndim"""
    max_ndim = max([ tensor.ndim for tensor in tensors ])
    return [
        tensor.view(*([1]*(max_ndim-tensor.ndim)), *tensor.shape) 
        for tensor in tensors
    ]

def log_vandermonde(
    v: Complex[Tensor, "*#C N"], 
    logx: Complex[Tensor, "*#C N"], 
    L: int,
) -> Float[Tensor, "*#C L"]:
    """
    computes a Vandermonde matrix-vector product.

    equivalent to:
    > vandermonde_matrix = th.exp(logx[...,None] * th.arange(L).to(logx))
    > mv_prod = th.einsum('...n,...nl->...l', v, vandermonde_matrix)
    > return 2*mv_prod.real
    """
    l = th.arange(L).to(logx)
    v,logx,l = broadcast_dims(v,logx,l)

    p: Float[Tensor, "H L 2"] = Genred(
        'ComplexMult(v, ComplexExp(ComplexMult(x, l)))',
        [ 'v = Vj(2)', 'x = Vj(2)', 'l = Vi(2)' ],
        reduction_op='Sum',
        axis=1,
    )(th.view_as_real(v), th.view_as_real(logx), th.view_as_real(l)) # type: ignore

    # return twice the real part, effectively includes conjugate pair parameters in the sum reduction
    return 2 * p[...,0]


@dataclass
class S4Args:
    state_size: int = 64
    dt_min: float = .001
    dt_max: float = .1
    bidirectional: bool = True

class S4D(nn.Module):
    """
    reimplementation of S4D (https://arxiv.org/abs/2206.11893)

    sources:
    - https://srush.github.io/annotated-s4/s4d
    - https://srush.github.io/annotated-s4/
    - https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py
    - https://github.com/state-spaces/s4/blob/main/models/s4/s4.py
    """
        
    def __init__(
        self, 
        H: int, 
        args: S4Args,
    ):
        super().__init__()

        # diagonalizing real SSM results in conjugate pair parameters -
        # state size can be halved if taken into account (see `log_vandermonde` return) 
        N = args.state_size // 2

        log_dt = np.log(args.dt_min) + th.rand(H) * (np.log(args.dt_max) - np.log(args.dt_min))
        self.log_dt = nn.Parameter(log_dt)
        setattr(self.log_dt, 'opt_key', 's4')

        # S4D-Inv initialization approximates S4D-LegS using inverse-law
        A_re = th.ones(N) * -.5
        A_im = N*2/th.pi * (N*2/(th.arange(N) * 2 + 1) - 1)

        self.log_neg_A_re = nn.Parameter(repeat(th.log(-A_re), 'n -> h n', h=H).clone())
        setattr(self.log_neg_A_re, 'opt_key', 's4')
        self.A_im = nn.Parameter(repeat(A_im, 'n -> h n', h=H).clone())
        setattr(self.A_im, 'opt_key', 's4')

        self.B = nn.Parameter(th.view_as_real(th.ones(H, N, dtype=th.cfloat)))
        setattr(self.B, 'opt_key', 's4')

        C = 2 if args.bidirectional else 1
        self.C = nn.Parameter(th.view_as_real(th.randn(C, H, N, dtype=th.cfloat)))

        self.D = nn.Parameter(th.randn(H))
        

    def forward(self, u: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L"]:
        L = u.size(-1)

        dt = th.exp(self.log_dt)
        C = th.view_as_complex(self.C) * th.view_as_complex(self.B)
        A = -th.exp(self.log_neg_A_re) + 1j * self.A_im

        dtA = dt[...,None] * A # H N
        C = C * th.expm1(dtA) / A
        K = log_vandermonde(C, dtA, L) # C * exp(dtA) ^ [0..L-1]

        if K.size(0) == 2:
            # bidirectional
            k0, k1 = K
            K = F.pad(k0, (0, L)) + F.pad(k1.flip(-1), (L, 0))
        else:
            K = K[0]

        # convolution
        k_f = th.fft.rfft(K.float(), n=2*L) # H L
        u_f = th.fft.rfft(u.float(), n=2*L) # B H L
        y = th.fft.irfft(u_f*k_f, n=2*L)[..., :L] # B H L

        y = y + u * self.D[:,None]

        return y.type_as(u)


class S4Block(nn.Module):
    """Gated State Space layer"""

    def __init__(self, dim: int, args: S4Args):
        super().__init__()

        self.proj_in = nn.Sequential(
            nn.Conv1d(dim, dim, 5,1,2, groups=dim),
            nn.Conv1d(dim, 2*dim, 1),
            nn.GroupNorm(2, 2*dim),
            nn.SiLU(),
        )

        self.seq = nn.Sequential(
            S4D(dim, args),
            nn.Conv1d(dim, dim, 1),
        )

        self.proj_out = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.GroupNorm(1, dim),
        )

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L"]:
        gate, h = self.proj_in(x).chunk(2, dim=1)
        return self.proj_out(gate * self.seq(h))