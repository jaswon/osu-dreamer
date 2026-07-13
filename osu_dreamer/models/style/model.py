
from dataclasses import dataclass
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.data.beatmap.encode import NUM_LABELS

from osu_dreamer.common.fourier_features import FourierFeatures

def zero(m: nn.Linear):
    nn.init.zeros_(m.weight)
    nn.init.zeros_(m.bias)
    return m

@dataclass
class StyleModelArgs:
    noise_level_features: int
    h_dim: int
    depth: int
    expand: int

class StyleModel(nn.Module):
    def __init__(self, style_dim: int, args: StyleModelArgs):
        super().__init__()
        self.style_dim = style_dim
        
        self.rff = FourierFeatures(1, args.noise_level_features, n_bins=32)
        self.proj_time = nn.Linear(args.noise_level_features, args.h_dim)
        self.cond_proj_w = nn.Parameter(th.empty(NUM_LABELS, args.noise_level_features, args.h_dim))
        self.cond_proj_b = nn.Parameter(th.zeros(NUM_LABELS, args.h_dim))
        for w in self.cond_proj_w:
            nn.init.xavier_uniform_(w)
        self.null_labels = nn.Parameter(th.randn(NUM_LABELS, args.h_dim) * args.h_dim ** -.5)

        self.proj_in = nn.Linear(style_dim, args.h_dim)
        self.proj_out = nn.Sequential(nn.RMSNorm(args.h_dim), nn.Linear(args.h_dim, style_dim))

        self.films = nn.ModuleList([
            zero(nn.Linear(args.h_dim, 2*args.h_dim))
            for _ in range(args.depth)
        ])
        self.norms = nn.ModuleList([
            nn.RMSNorm(args.h_dim)
            for _ in range(args.depth)
        ])
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(args.h_dim, args.expand * args.h_dim),
                nn.SiLU(),
                nn.Linear(args.expand * args.h_dim, args.h_dim),
                nn.RMSNorm(args.h_dim),
            )
            for _ in range(args.depth)
        ])

    def compute_conditioning(
        self,
        labels: Float[Tensor, str(f"B {NUM_LABELS}")], # [0,10]
        t: Float[Tensor, "B"], # [0,1]
    ) -> Float[Tensor, "B H"]:
        labels = labels[:,:,None] # B N 1
        h = th.einsum('bnf,nfh->bnh', self.rff(labels/10), self.cond_proj_w) + self.cond_proj_b # B N H
        h = th.where(labels < 0, self.null_labels[None], h)
        return h.sum(dim=1) + self.proj_time(self.rff(t[:,None]))

    def forward(
        self,
        ut: Float[Tensor, "B S"],     # noised style code
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
        t: Float[Tensor, "B"],        # noise level
    ) -> Float[Tensor, "B S"]:
        c = self.compute_conditioning(labels, t)
        h = self.proj_in(ut)
        for film, norm, block in zip(self.films, self.norms, self.blocks):
            scale, shift = film(c).chunk(2, dim=1)
            h = h + block(norm(h) * (1 + scale) + shift)
        return self.proj_out(h)

    @th.no_grad()
    def sample(
        self,
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
        num_steps: int = 16,
    ) -> Float[Tensor, "B S"]:
        u = th.randn(labels.size(0), self.style_dim, device=labels.device)
        ts = th.linspace(0, 1, num_steps+1, device=labels.device)
        for t0, t1 in zip(ts[:-1], ts[1:]):
            u = u + self(u, labels, t0.expand(labels.size(0))) * (t1 - t0)
        return u
