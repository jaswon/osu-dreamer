
from dataclasses import dataclass
from math import sqrt
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from osu_dreamer.data.beatmap.encode import NUM_LABELS

from osu_dreamer.common.rms_norm import rms_norm
from osu_dreamer.common.fourier_features import FourierFeatures

def zero(m: nn.Linear):
    nn.init.zeros_(m.weight)
    nn.init.zeros_(m.bias)
    return m

@dataclass
class StyleModelArgs:
    label_features: int
    h_dim: int
    depth: int
    expand: int
    dropout: float = 0.

class StyleModel(nn.Module):
    def __init__(self, style_dim: int, args: StyleModelArgs):
        super().__init__()
        self.style_dim = style_dim

        # distance field constants: style codes are RMS-normalized (||s||^2 = S),
        # so E[d^2] between N(0,I) noise and data is exactly 2S.
        d0_sq = 2. * style_dim
        # noise floor: squared distance at the 99th percentile of the
        # logit-normal time distribution used in training
        t99 = th.tensor(2.3263478740408408).sigmoid().item() # sigmoid(ndtri(.99))
        self.c0 = (1 - t99)**2 * d0_sq
        self.u_scale = sqrt(d0_sq)

        self.rff = FourierFeatures(1, args.label_features, n_bins=32)
        self.cond_proj_w = nn.Parameter(th.empty(NUM_LABELS, args.label_features, args.h_dim))
        self.cond_proj_b = nn.Parameter(th.zeros(NUM_LABELS, args.h_dim))
        for w in self.cond_proj_w:
            nn.init.xavier_uniform_(w)
        self.null_labels = nn.Parameter(th.randn(NUM_LABELS, args.h_dim) * args.h_dim ** -.5)

        self.proj_in = nn.Linear(style_dim, args.h_dim)
        self.proj_out = nn.Sequential(nn.RMSNorm(args.h_dim), zero(nn.Linear(args.h_dim, style_dim)))

        # distance head: shares the trunk, reads the final hidden state
        self.u_out = nn.Linear(args.h_dim, 1)
        nn.init.zeros_(self.u_out.weight)
        # init predicted distance to its marginal mean: E[1-t]*sqrt(2S) = .5*u_scale
        # => softplus(bias) = .5 => bias = log(exp(.5)-1)
        nn.init.constant_(self.u_out.bias, -0.4328) # type: ignore

        self.films = nn.ModuleList([
            zero(nn.Linear(args.h_dim, 3*args.h_dim))
            for _ in range(args.depth)
        ])
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.h_dim, args.expand * args.h_dim),
                nn.SiLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.expand * args.h_dim, args.h_dim),
            )
            for _ in range(args.depth)
        ])

    def compute_conditioning(
        self,
        labels: Float[Tensor, str(f"B {NUM_LABELS}")], # [0,10]
    ) -> Float[Tensor, "B H"]:
        labels = labels[:,:,None] # B N 1
        h = th.einsum('bnf,nfh->bnh', self.rff(labels/10), self.cond_proj_w) + self.cond_proj_b # B N H
        h = th.where(labels < 0, self.null_labels[None], h)
        return h.sum(dim=1)

    def forward(
        self,
        st: Float[Tensor, "B S"],     # noised style code
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
    ) -> tuple[
        Float[Tensor, "B"],  # u: distance to the style manifold
        Float[Tensor, "B S"],# v: direction away from the style manifold
    ]:
        c = self.compute_conditioning(labels)
        x = self.proj_in(st)
        for film, block in zip(self.films, self.blocks):
            scale, shift, gate = film(c).chunk(3, dim=1)
            h = rms_norm(x) * (1 + scale) + shift
            h = block(h)
            h = rms_norm(h) * gate
            x = x + h
        v = self.proj_out(x)
        u = self.u_scale * F.softplus(self.u_out(rms_norm(x))).squeeze(-1)
        return u, v

    @th.no_grad()
    def sample(
        self,
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
        num_steps: int = 16,
    ) -> Float[Tensor, "B S"]:
        s = th.randn(labels.size(0), self.style_dim, device=labels.device)

        # sphere tracing with a self-calibrating step size: contract the
        # predicted distance geometrically from its initial value down to the
        # field's noise floor sqrt(c0) over the step budget.
        u0 = self(s, labels)[0].mean().item()
        eta = 1. - (sqrt(self.c0) / max(u0, sqrt(self.c0) + 1e-6)) ** (1. / num_steps)

        for _ in range(num_steps):
            u, v = self(s, labels)
            s = s - eta * u[:,None] * v
        return s
