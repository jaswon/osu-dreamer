"""
reimplementation of S4D (https://arxiv.org/abs/2206.11893)

sources:
- https://srush.github.io/annotated-s4/s4d
- https://srush.github.io/annotated-s4/
- https://github.com/state-spaces/s4/blob/main/models/s4/s4d.py
- https://github.com/state-spaces/s4/blob/main/models/s4/s4.py
"""

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


def make_DPLR_HiPPO(N: int) -> tuple[
    Float[Tensor, "N"],   # Lambda_real
    Float[Tensor, "N"],   # Lambda_imag
]:
    """diagonalized HiPPO matrix"""

    # make NxN HiPPO matrix
    P = th.sqrt(1 + 2 * th.arange(N))
    A = P[:, None] * P[None, :]
    A = th.diag(th.arange(N)) - th.tril(A)

    # construct NPLR representation
    P = th.sqrt(th.arange(N) + 0.5) # Add in a rank 1 term. Makes it Normal.
    S = A + P[:, None] * P[None, :]

    # diagonalize the NPLR representation
    S_diag = th.diagonal(S)
    Lambda_real = th.mean(S_diag) * th.ones_like(S_diag)
    Lambda_imag = th.linalg.eigvalsh(S * -1j)

    return Lambda_real, Lambda_imag


class S4D(nn.Module):
    def __init__(
        self, 
        H: int, 
        N: int = 64, 
        dt_min: float = .001, 
        dt_max: float = .1,
        bidirectional: bool = True,
        initialization: str = 'inv',
    ):
        super().__init__()

        # diagonalizing real SSM results in conjugate pair parameters -
        # state size can be halved if taken into account (see `log_vandermonde` return) 
        N = N // 2

        self.bidirectional = bidirectional
        C = 2 if bidirectional else 1

        log_dt = np.log(dt_min) + th.rand(H) * (np.log(dt_max) - np.log(dt_min))
        self.log_dt = nn.Parameter(log_dt)
        setattr(self.log_dt, '_s4_optim', True)

        if initialization == 'legs':
            # TODO: deduplicate conjugate pairs
            A_re, A_im = make_DPLR_HiPPO(N)
        elif initialization == 'lin':
            A_re = th.ones(N) * -.5
            A_im = th.arange(N) * th.pi
        elif initialization == 'inv':
            A_re = th.ones(N) * -.5
            n2p1 = th.arange(N) * 2 + 1
            A_im = N / th.pi * (N/n2p1 - 1)
        else:
            raise NotImplementedError(f'unknown initialization `{initialization}`')

        assert (A_re<0).all(), '`A_re` should be negative'
        self.log_neg_A_re = nn.Parameter(repeat(th.log(-A_re), 'n -> h n', h=H).clone())
        setattr(self.log_neg_A_re, '_s4_optim', True)
        self.A_im = nn.Parameter(repeat(A_im, 'n -> h n', h=H).clone())
        setattr(self.A_im, '_s4_optim', True)

        self.B = nn.Parameter(th.view_as_real(th.ones(H, N, dtype=th.cfloat)))
        setattr(self.B, '_s4_optim', True)

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

        if self.bidirectional:
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
    def __init__(self, dim: int, expand: int = 1):
        super().__init__()

        h_dim = dim * expand if expand > 0 else dim

        self.proj_in = nn.Conv1d(dim, h_dim*2, 1)
        self.seq = S4D(h_dim)
        self.proj_out = nn.Sequential(
            nn.GroupNorm(1, h_dim),
            nn.Conv1d(h_dim, dim, 1),
        )

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L"]:
        gate, h = self.proj_in(x).chunk(2, dim=1)
        return self.proj_out(F.silu(gate) * self.seq(h))