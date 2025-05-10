
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
from osu_dreamer.modules.lr_schedule import make_lr_schedule, LRScheduleArgs

from .diffusion import Diffusion, DiffusionArgs
from .denoiser import Denoiser, DenoiserArgs

    
class Model(pl.LightningModule):
    def __init__(
        self,

        # validation parameters
        val_batches: int,
        val_steps: int,

        # training parameters
        opt_args: dict[str, Any],
        lr_schedule: LRScheduleArgs,

        # model hparams
        diffusion_args: DiffusionArgs,
        denoiser_args: DenoiserArgs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # model
        self.diffusion = Diffusion(diffusion_args)
        self.denoiser = Denoiser(
            X_DIM, 
            A_DIM, 
            diffusion_args.noise_level_features, 
            denoiser_args,
        )

        # self.diffusion.compile()
        # self.denoiser.compile()

        # validation params
        self.val_batches = val_batches
        self.val_steps = val_steps

        # training params
        self.opt_args = opt_args
        self.lr_schedule = make_lr_schedule(lr_schedule)
    

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        chart: Float[Tensor, str(f"B {X_DIM} L")],
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:

        denoiser = partial(self.denoiser,audio,self.preprocess_labels(labels))
        pred_chart, u, loss_weight = self.diffusion.training_sample(denoiser, chart)

        pixel_loss = loss_weight * (pred_chart - chart).pow(2).mean((1,2))
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

        z_a = repeat(audio[None], '1 a l -> b a l', b=num_samples)
        z = th.randn(num_samples, X_DIM, z_a.size(-1), device=audio.device)
        return self.diffusion.sample(
            partial(self.denoiser,z_a,self.preprocess_labels(labels)), 
            num_steps, z,
            **kwargs,
        )

#
#
# =============================================================================
# MODEL TRAINING
# =============================================================================
#
#

    def configure_optimizers(self):
        opt = AdaBelief(self.parameters(), **self.opt_args)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": th.optim.lr_scheduler.LambdaLR(opt, self.lr_schedule),
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

    @th.no_grad()
    def plot_sample(self, b: Batch):
        a, x, label = b
        pred_x = self.sample(a[0], label)

        exp: SummaryWriter = self.logger.experiment # type: ignore
        with plot_signals(
            a[0].cpu().numpy(),
            [ x[0].cpu().numpy() for x in [ x, pred_x ] ],
        ) as fig:
            exp.add_figure("samples", fig, global_step=self.global_step)
        