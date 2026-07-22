
from functools import partial
from math import sqrt

from dataclasses import dataclass
from jaxtyping import Float

import torch as th
from torch import Tensor, nn
import torch.nn.functional as F

import tqdm

from .backbone import Backbone, BackboneArgs, zero

@dataclass
class DiffusionModelArgs:
    global_cond_dim: int
    backbone_dim: int
    backbone_args: BackboneArgs
    u_head_dim: int = 64

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

        # distance field constants, in the per-frame metric.
        # E[d^2] between N(0,I) noise and per-frame RMS-normalized latents is 2E.
        d0_sq = 2. * emb_dim
        # c0 smooths the field near the manifold: u -> sqrt(d^2 + c0).
        # set to the squared distance at the 99th percentile of the logit-normal
        # time distribution used in training - the finest resolvable noise level.
        t99 = th.tensor(2.3263478740408408).sigmoid().item() # sigmoid(ndtri(.99))
        self.c0 = (1 - t99)**2 * d0_sq
        self.u_scale = sqrt(d0_sq)

        self.proj_audio = nn.Sequential(nn.Conv1d(a_dim, a_dim, 1), nn.SiLU())
        self.proj_style = nn.Sequential(nn.Linear(style_dim, args.global_cond_dim), nn.SiLU())

        self.proj_in = nn.Conv1d(emb_dim, args.backbone_dim, 1)
        self.net = Backbone(args.backbone_dim, a_dim, args.global_cond_dim, args.backbone_args)
        self.proj_out = nn.Conv1d(args.backbone_dim, emb_dim, 1)
        nn.init.zeros_(self.proj_out.weight)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)

        # lightweight distance head on the raw noised input (paper Sec. 5.3),
        # modulated by the global conditioning.
        U = args.u_head_dim
        self.u_head = nn.Sequential(
            nn.Conv1d(emb_dim, emb_dim, 3,1,1, groups=emb_dim),
            nn.Conv1d(emb_dim, U, 1), 
            nn.SiLU(),
            nn.Conv1d(U, U, 3,1,1, groups=U),
            nn.Conv1d(U, U, 1), 
            nn.SiLU(),
        )
        self.u_mod = zero(nn.Linear(args.global_cond_dim, 2*U))
        self.u_out = nn.Linear(U, 1)
        nn.init.zeros_(self.u_out.weight)
        # init predicted distance to its marginal mean: E[1-t]*sqrt(2E) = .5*u_scale
        # => softplus(bias) = .5 => bias = log(exp(.5)-1)
        nn.init.constant_(self.u_out.bias, -0.4328) # type: ignore

    def _precompute_conditioning(
        self,
        audio: Float[Tensor, "B A l"],
        style: Float[Tensor, "#B S"],
    ) -> tuple[
        Float[Tensor, "#B A l"],
        Float[Tensor, "B C"],
    ]:
        return (
            self.proj_audio(audio),
            self.proj_style(style),
        )
    
    def _pred(
        self,
        a: Float[Tensor, "#B A l"],
        cg: Float[Tensor, "B C"],
        xt: Float[Tensor, "B E l"],
    ) -> tuple[
        Float[Tensor, "B"],     # u: per-frame distance to the data manifold
        Float[Tensor, "B E l"], # v: direction away from the data manifold
    ]:
        h = self.proj_in(xt)
        h = self.net(h,cl=a,cg=cg)
        v = self.proj_out(h)

        f = self.u_head(xt).mean(-1)
        scale, shift = self.u_mod(cg).chunk(2, dim=-1)
        f = f * (1 + scale) + shift
        u = self.u_scale * F.softplus(self.u_out(f)).squeeze(-1)
        return u, v

    def forward(
        self, 
        audio: Float[Tensor, "B A l"],
        style: Float[Tensor, "B S"],
        xt: Float[Tensor, "B E l"],
    ) -> tuple[
        Float[Tensor, "B"],
        Float[Tensor, "B E l"],
    ]:
        return self._pred(*self._precompute_conditioning(audio, style), xt)
        
    
    @th.no_grad()
    def sample(
        self, 
        audio: Float[Tensor, "#B A l"],
        style: Float[Tensor, "B S"],
        num_steps: int,
        show_progress: bool = False,
    ) -> Float[Tensor, "B E l"]:
        x = th.randn(style.size(0), self.emb_dim, audio.size(-1), device=audio.device)
        pred = partial(self._pred, *self._precompute_conditioning(audio, style))

        # sphere tracing with a self-calibrating step size: contract the
        # predicted distance geometrically from its initial value down to the
        # field's noise floor sqrt(c0) over the step budget.
        u0 = pred(x)[0].mean().item()
        eta = 1. - (sqrt(self.c0) / max(u0, sqrt(self.c0) + 1e-6)) ** (1. / num_steps)

        for _ in tqdm.trange(num_steps, disable=not show_progress):
            u, v = pred(x)
            x = x - eta * u[:,None,None] * v

        return x