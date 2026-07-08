
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.data.beatmap.encode import NUM_LABELS, X_DIM
from osu_dreamer.data.load_audio import A_DIM
from osu_dreamer.data.module import pad_to_multiple

from osu_dreamer.models.diffusion.model import DiffusionModel, DiffusionModelArgs
from osu_dreamer.models.diffusion.style_prior import StylePrior, StylePriorArgs
from osu_dreamer.models.latent.model import LatentModel, LatentModelArgs

@dataclass
class LDMArgs:
    emb_dim: int
    style_dim: int
    n_downs: int
    stride: int
    latent_args: LatentModelArgs
    style_prior_args: StylePriorArgs
    diffusion_args: DiffusionModelArgs

class LDM(nn.Module):
    def __init__(self, args: LDMArgs):
        super().__init__()
        self.latent = LatentModel(args.emb_dim, args.style_dim, args.n_downs, args.stride, args.latent_args)
        self.style_prior = StylePrior(args.style_dim, args.style_prior_args)
        self.diffusion = DiffusionModel(args.emb_dim, args.latent_args.h_dim, args.style_dim, args.diffusion_args)

    @th.no_grad()
    def sample(
        self, 
        audio: Float[Tensor, str(f"{A_DIM} L")],
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
        num_steps: int,
        show_progress: bool = False,
    ) -> tuple[
        Float[Tensor, str(f"B {X_DIM} L")], 
        Float[Tensor, str(f"B {NUM_LABELS}")],
    ]:
        L = audio.size(-1)
        audio = pad_to_multiple(audio, self.latent.chunk_size)

        skips, h = self.latent.audio_encoder(audio[None])
        s = self.style_prior.sample(labels)
        z = self.diffusion.sample(h, labels, s, num_steps, show_progress=show_progress)
        chart, out_labels = self.latent.decode(z, s, skips=skips)
        return chart[..., :L], out_labels