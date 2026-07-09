
from functools import partial

from dataclasses import dataclass
from jaxtyping import Float

import torch as th
from torch import Tensor, nn

import tqdm

from osu_dreamer.data.beatmap.encode import NUM_LABELS

from osu_dreamer.common.fourier_features import FourierFeatures

from .backbone import Backbone, BackboneArgs

@dataclass
class DiffusionModelArgs:
    noise_level_features: int
    global_cond_dim: int
    
    backbone_dim: int
    backbone_args: BackboneArgs

class DiffusionModel(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        a_dim: int,
        style_dim: int,
        args: DiffusionModelArgs,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.style_dim = style_dim

        self.proj_time = nn.Sequential(
            FourierFeatures(1, args.noise_level_features),
            nn.Linear(args.noise_level_features, args.global_cond_dim),
        )
        self.proj_label = nn.Linear(NUM_LABELS, args.global_cond_dim)
        self.proj_style = nn.Linear(style_dim, args.global_cond_dim)

        self.proj_in = nn.Conv1d(emb_dim+a_dim, args.backbone_dim, 1)
        self.net = Backbone(args.backbone_dim, a_dim, args.global_cond_dim, args.backbone_args)
        self.proj_out = nn.Conv1d(args.backbone_dim, emb_dim, 1)
        nn.init.zeros_(self.proj_out.weight)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)

    def _precompute_conditioning(
        self,
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
        style: Float[Tensor, "#B S"],
    ) -> Float[Tensor, "B C"]:
        return self.proj_label(labels) + self.proj_style(style)
    
    def _pred_flow(
        self,
        cg: Float[Tensor, "B C"],
        a: Float[Tensor, "#B A l"],
        xt: Float[Tensor, "B E l"], # noised input
        t: Float[Tensor, "#B"],     # noise level
    ) -> Float[Tensor, "B E l"]:
        cg = cg + self.proj_time(t[:,None])
        h = self.proj_in(th.cat([xt, a.expand(xt.size(0), -1, -1)], dim=1))
        h = self.net(h,cl=a,cg=cg)
        return self.proj_out(h)

    def forward(
        self, 
        audio: Float[Tensor, "B A l"],
        labels: Float[Tensor, "B C"],
        style: Float[Tensor, "B S"],
        
        # --- diffusion args --- #
        xt: Float[Tensor, "B E l"], # noised input
        t: Float[Tensor, "B"],      # noise level
    ) -> Float[Tensor, "B E l"]:
        return self._pred_flow(self._precompute_conditioning(labels, style), audio, xt, t)
        
    
    @th.no_grad()
    def sample(
        self, 
        audio: Float[Tensor, "#B A l"],
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
        style: Float[Tensor, "B S"],
        num_steps: int,
        time_shift: float = 3.,
        show_progress: bool = False,
    ) -> Float[Tensor, "B E l"]:
        B = labels.size(0)
        x = th.randn(B, self.emb_dim, audio.size(-1), device=audio.device)
        denoiser = partial(
            self._pred_flow,
            self._precompute_conditioning(labels, style),
            audio,
        )

        # shifted timestep schedule: denser steps near t=1
        u = th.linspace(0, 1, num_steps+1, device=audio.device)[:,None]
        ts = time_shift * u / (1 + (time_shift - 1) * u)

        # heun step
        for t0, t1 in tqdm.tqdm(
            list(zip(ts[:-1], ts[1:])),
            disable=not show_progress,
        ):
            dt = t1 - t0
            v0 = denoiser(x, t0)
            v1 = denoiser(x + v0 * dt, t1)
            x = x + 0.5 * (v0 + v1) * dt

        return x