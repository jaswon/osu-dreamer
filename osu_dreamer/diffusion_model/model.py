
from functools import partial

from dataclasses import dataclass
from jaxtyping import Float

import torch as th
from torch import Tensor, nn

from einops import repeat

import tqdm

from osu_dreamer.data.load_audio import A_DIM
from osu_dreamer.data.beatmap.encode import NUM_LABELS

from osu_dreamer.modules.spec_features import SpecFeatures
from osu_dreamer.modules.ae import AEArgs, Encoder
from osu_dreamer.modules.fourier_features import FourierFeatures

from .backbone import Backbone, BackboneArgs

@dataclass
class DiffusionModelArgs:
    n_audio_features: int
    backbone_dim: int
    global_cond_dim: int
    global_cond_h: int
    noise_level_features: int
    backbone_args: BackboneArgs
    audio_encoder_args: AEArgs

class DiffusionModel(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        n_downs: int,
        stride: int,
        flow_latent_dim: int,
        args: DiffusionModelArgs,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.chunk_size = stride ** n_downs
        self.flow_latent_dim = flow_latent_dim

        self.proj_audio = nn.Sequential(
            SpecFeatures(A_DIM, args.n_audio_features),
            Encoder(args.n_audio_features, args.n_audio_features, n_downs, stride, args.audio_encoder_args),
        )

        self.proj_time = nn.Sequential(
            FourierFeatures(1, args.noise_level_features),
            nn.Linear(args.noise_level_features, args.global_cond_h, bias=False),
        )
        self.proj_label = nn.Linear(NUM_LABELS, args.global_cond_h, bias=False)
        self.proj_latent = nn.Linear(flow_latent_dim, args.global_cond_h)

        self.proj_cond = nn.Sequential(
            nn.SiLU(),
            nn.Linear(args.global_cond_h, args.global_cond_dim, bias=False),
            nn.SiLU(),
        )

        self.proj_in = nn.Conv1d(emb_dim, args.backbone_dim, 1)
        self.net = Backbone(args.backbone_dim, args.n_audio_features, args.global_cond_dim, args.backbone_args)
        self.proj_out = nn.Conv1d(args.backbone_dim, emb_dim, 1)
        nn.init.zeros_(self.proj_out.weight)
        if self.proj_out.bias is not None:
            nn.init.zeros_(self.proj_out.bias)

    def _precompute_conditioning(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        labels: Float[Tensor, "B C"],
        latent: Float[Tensor, "B Z"],
    ) -> tuple[
        Float[Tensor, "B Cl l"],
        Float[Tensor, "B Cg"],
    ]:
        return (
            self.proj_audio(audio),
            self.proj_label(labels) + self.proj_latent(latent),
        )
    
    def _pred_flow(
        self,
        cond_l: Float[Tensor, "B Cl l"],
        cond_g: Float[Tensor, "B Cg"],
        xt: Float[Tensor, "B E l"], # noised input
        t: Float[Tensor, "B"],      # noise level
    ) -> Float[Tensor, "B E l"]:
        cond_g = self.proj_cond( cond_g + self.proj_time(t[:,None]) )
        h = self.proj_in(xt)
        h = self.net(h,cond_l=cond_l,cond_g=cond_g)
        return self.proj_out(h)

    def forward(
        self, 
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        labels: Float[Tensor, "B C"],
        latent: Float[Tensor, "B Z"], # flow latent
        
        # --- diffusion args --- #
        xt: Float[Tensor, "B E l"], # noised input
        t: Float[Tensor, "B"],      # noise level
    ) -> Float[Tensor, "B E l"]:
        return self._pred_flow(*self._precompute_conditioning(audio, labels, latent), xt, t)
        
    
    @th.no_grad()
    def sample(
        self, 
        audio: Float[Tensor, str(f"{A_DIM} L")],
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
        num_steps: int,
        show_progress: bool = False,
        schedule_std: float = 1.,
    ) -> Float[Tensor, "B E l"]:
        num_samples = labels.size(0)

        # determine size of noise sample
        L = 1 + (audio.size(-1)-1) // self.chunk_size

        x = th.randn(num_samples, self.emb_dim, L, device=audio.device)
        denoiser = partial(
            self._pred_flow,
            *self._precompute_conditioning(
                repeat(audio[None], '1 a l -> b a l', b=num_samples),
                labels,
                th.randn(num_samples, self.flow_latent_dim, device=audio.device),
            )
        )

        # non-uniform time schedule: warp a uniform grid through the inverse CDF of
        # the logit-normal(0, schedule_std) distribution used during training, placing
        # smaller steps near t=0.5 (where the model is trained most) and larger steps
        # near the extremes. schedule_std=1 matches the training distribution exactly;
        # as schedule_std -> inf the schedule approaches equidistant steps.
        u = th.linspace(0, 1, num_steps + 1, device=audio.device)
        t_nodes = (schedule_std * 2 ** 0.5 * th.special.erfinv(2 * u - 1)).sigmoid()
        dt = t_nodes[1:] - t_nodes[:-1]

        # euler step
        for t, step in tqdm.tqdm(
            zip(repeat(t_nodes[:-1], 'n -> n b', b=x.size(0)), dt),
            total=num_steps,
            disable=not show_progress,
        ):
            x = x + denoiser(x, t) * step

        return x