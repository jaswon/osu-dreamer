
from dataclasses import dataclass
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.data.beatmap.encode import NUM_LABELS

from osu_dreamer.modules.fourier_features import FourierFeatures

@dataclass
class StylePriorArgs:
    noise_level_features: int
    h_dim: int
    depth: int

class StylePrior(nn.Module):
    """flow matching over style codes: fits the aggregate style posterior q(s | labels)
    so inference can sample real-looking styles instead of relying on q(s) = N(0,I)"""
    def __init__(
        self,
        style_dim: int,
        args: StylePriorArgs,
    ):
        super().__init__()
        self.style_dim = style_dim
        self.proj_time = nn.Sequential(
            FourierFeatures(1, args.noise_level_features),
            nn.Linear(args.noise_level_features, args.h_dim),
        )
        self.proj_cond = nn.Linear(style_dim + NUM_LABELS, args.h_dim)
        layers: list[nn.Module] = []
        for _ in range(args.depth):
            layers.extend([ nn.SiLU(), nn.Linear(args.h_dim, args.h_dim) ])
        self.net = nn.Sequential(*layers, nn.SiLU(), nn.Linear(args.h_dim, style_dim))

    def forward(
        self,
        ut: Float[Tensor, "B S"],     # noised style code
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
        t: Float[Tensor, "B"],        # noise level
    ) -> Float[Tensor, "B S"]:
        h = self.proj_cond(th.cat([ut, labels], dim=1)) + self.proj_time(t[:,None])
        return self.net(h)

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
