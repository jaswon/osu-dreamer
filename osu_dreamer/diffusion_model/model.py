
from functools import partial

from typing import Any
from jaxtyping import Float

import torch as th
from torch import Tensor

from einops import repeat, rearrange

import pytorch_lightning as pl
from torch.utils.tensorboard.writer import SummaryWriter

from osu_dreamer.data.dataset import Batch
from osu_dreamer.data.load_audio import A_DIM
from osu_dreamer.data.beatmap.encode import X_DIM
from osu_dreamer.data.labels import NUM_LABELS
from osu_dreamer.data.plot import plot_signals

from osu_dreamer.modules.adabelief import AdaBelief

from osu_dreamer.latent_model.model import Model as LatentModel
from .diffusion import Diffusion, DiffusionArgs
from .denoiser import Denoiser, DenoiserArgs
from .audio_features import AudioFeatures, AudioFeatureArgs

    
class Model(pl.LightningModule):
    def __init__(
        self,

        # validation parameters
        val_batches: int,
        val_steps: int,

        # training parameters
        opt_args: dict[str, Any],
        step_ref: float,

        # model hparams
        latent_ckpt: str,
        diffusion_args: DiffusionArgs,
        denoiser_args: DenoiserArgs,
        audio_feature_args: AudioFeatureArgs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # model
        self.latent = LatentModel.load_from_checkpoint(latent_ckpt)
        self.latent.freeze()
        self.diffusion = Diffusion(diffusion_args)
        self.audio_features = AudioFeatures(self.latent.a_dim, audio_feature_args)
        self.denoiser = Denoiser(
            self.latent.x_dim, 
            audio_feature_args.dim, 
            diffusion_args.noise_level_features, 
            denoiser_args,
        )

        # validation params
        self.val_batches = val_batches
        self.val_steps = val_steps

        # training params
        self.opt_args = opt_args
        self.step_ref = step_ref
    

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        chart: Float[Tensor, str(f"B {X_DIM} L")],
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        
        with th.no_grad():
            z_x = self.latent.chart_encoder(chart)
            z_a = self.latent.audio_encoder(audio)

        denoiser = partial(self.denoiser,self.audio_features(z_a),labels)
        pred_z_x, u, loss_weight = self.diffusion.training_sample(denoiser, z_x)

        pixel_loss = loss_weight * (pred_z_x - z_x).pow(2).mean((1,2))
        loss = (pixel_loss / th.exp(u) + u).mean()
        return loss, {
            "loss": loss.detach(),
            "pixel": pixel_loss.detach().mean(),
        }
    
    @th.no_grad()
    def sample(
        self, 
        audio: Float[Tensor, str(f"{A_DIM} L")],
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
        num_steps: int = 0,
        **kwargs,
    ) -> Float[Tensor, str(f"B {X_DIM} L")]:
        num_steps = num_steps if num_steps > 0 else self.val_steps
        num_samples = labels.size(0)

        def make_latent(z_a: Float[Tensor, "A zL"]) -> Float[Tensor, "B X zL"]:
            z = th.randn(num_samples, self.latent.x_dim, z_a.size(-1), device=audio.device)
            a_f = repeat(self.audio_features(z_a[None]), '1 a l -> b a l', b=num_samples)
            return self.diffusion.sample(
                partial(self.denoiser,a_f,labels), 
                num_steps, z,
                **kwargs,
            )

        return self.latent.generate(audio, make_latent)

#
#
# =============================================================================
# MODEL TRAINING
# =============================================================================
#
#

    def configure_optimizers(self):
        opt = AdaBelief(self.parameters(), **self.opt_args)
        isqrt = lambda step: max(step / self.step_ref, 1) ** -.5
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": th.optim.lr_scheduler.LambdaLR(opt, isqrt),
                "interval": "step",
            }
        }

    def training_step(self, batch: Batch, batch_idx):
        loss, log_dict = self(*batch)
        self.log_dict({ f"train/{k}": v for k,v in log_dict.items() })
        return loss
 
    def validation_step(self, batch: Batch, batch_idx, *args, **kwargs):
        with th.no_grad():
            a,x,l = batch
            bL = self.val_batches * (a.size(-1) // self.val_batches)
            a = rearrange(a[...,:bL], '1 ... (b l) -> b ... l', b = self.val_batches)
            x = rearrange(x[...,:bL], '1 ... (b l) -> b ... l', b = self.val_batches)
            l = repeat(l, '1 d -> b d', b = self.val_batches)
            _, log_dict = self(a,x,l)
        self.log_dict({ f"val/{k}": v for k,v in log_dict.items() })

        if batch_idx == 0:
            self.plot_sample(batch)

    def plot_sample(self, b: Batch):
        a, x, label = b

        with th.no_grad():
            plots = [ x[0].cpu().numpy() for x in [ x, self.sample(a[0], label) ] ]

        with plot_signals(a[0].cpu().numpy(), plots) as fig:
            exp: SummaryWriter = self.logger.experiment # type: ignore
            exp.add_figure("samples", fig, global_step=self.global_step)