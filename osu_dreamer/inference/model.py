
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from osu_dreamer.data.beatmap.encode import NUM_LABELS, X_DIM
from osu_dreamer.data.load_audio import A_DIM

from osu_dreamer.diffusion_model.model import DiffusionModel, DiffusionModelArgs
from osu_dreamer.latent_model.model import LatentModel, LatentModelArgs

@dataclass
class LDMArgs:
    emb_dim: int
    n_downs: int
    stride: int
    flow_latent_dim: int
    latent_args: LatentModelArgs
    diffusion_args: DiffusionModelArgs

class LDM(nn.Module):
    def __init__(self, args: LDMArgs):
        super().__init__()
        self.latent = LatentModel(args.emb_dim, args.n_downs, args.stride, args.latent_args)
        self.diffusion = DiffusionModel(args.emb_dim, args.n_downs, args.stride, args.flow_latent_dim, args.diffusion_args)

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
        return self.latent.decode(audio, self.diffusion.sample(audio, labels, num_steps, show_progress))