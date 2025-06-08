
from functools import partial

from typing import Any
from jaxtyping import Float

import torch as th
from torch import Tensor

from einops import repeat, rearrange

import pytorch_lightning as pl
from torch.utils.tensorboard.writer import SummaryWriter
import tqdm

from osu_dreamer.data.load_audio import A_DIM
from osu_dreamer.data.beatmap.encode import X_DIM
from osu_dreamer.data.labels import NUM_LABELS
from osu_dreamer.data.plot import plot_signals

from osu_dreamer.modules.muon import Muon
from osu_dreamer.modules.lr_schedule import LRScheduleArgs, make_lr_schedule

from .data.dataset import Batch
from .denoiser import Denoiser, DenoiserArgs
from .encoder import Encoder, EncoderArgs

    
class Model(pl.LightningModule):
    def __init__(
        self,

        # validation parameters
        val_batches: int,
        val_steps: int,

        # training parameters
        grad_clip: float,
        opt_args: dict[str, Any],
        schedule_args: LRScheduleArgs,

        # model hparams
        a_h_dim: int,
        encoder_args: EncoderArgs,
        denoiser_args: DenoiserArgs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # validation params
        self.val_batches = val_batches
        self.val_steps = val_steps

        # training params
        self.grad_clip = grad_clip
        self.opt_args = opt_args
        self.lr_schedule = make_lr_schedule(schedule_args)

        # model
        self.audio_encoder = Encoder(A_DIM, a_h_dim, encoder_args)
        self.denoiser = Denoiser(X_DIM, a_h_dim, denoiser_args)
    
    def preprocess_labels(
        self, 
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
    ) -> Float[Tensor, str(f"B {NUM_LABELS}")]:
        return labels - 5

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        x1: Float[Tensor, str(f"B {X_DIM} L")],
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        
        B = audio.size(0)
        denoiser = partial(
            self.denoiser,
            self.audio_encoder(audio),
            self.preprocess_labels(labels),
        )
        
        x0 = th.randn_like(x1)
        true_flow = x1 - x0
        if self.training:
            # sample t from logit-normal distribution
            t = th.randn(B, device=x1.device).sigmoid()
        else:
            # sample t at evenly spaced points
            t = th.linspace(0, 1, B+2, device=x1.device)[1:-1]
        xt = th.lerp(x0,x1,t[:,None,None])
        pred_flow = denoiser(xt, t)
        loss = (true_flow - pred_flow).pow(2).mean()
        return loss, {
            "loss": loss.detach(),
        }
    
    @th.no_grad()
    def sample(
        self, 
        audio: Float[Tensor, str(f"{A_DIM} L")],
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
        num_steps: int = 0,
        show_progress: bool = False,
    ) -> Float[Tensor, str(f"B {X_DIM} L")]:
        num_steps = num_steps if num_steps > 0 else self.val_steps
        num_samples = labels.size(0)

        x = th.randn(num_samples, X_DIM, audio.size(-1), device=audio.device)
        denoiser = partial(
            self.denoiser,
            repeat(self.audio_encoder(audio[None]), '1 a l -> b a l', b=num_samples),
            self.preprocess_labels(labels),
        )
        for t in tqdm.tqdm(
            th.linspace(0, 1, num_steps, device=audio.device),
            disable=not show_progress,
        ):
            x = x + denoiser(x, t.expand(x.size(0))) / num_steps 
        return x.clamp(min=-1, max=1)

#
#
# =============================================================================
# MODEL TRAINING
# =============================================================================
#
#

    def configure_optimizers(self):
        opt = Muon(self.parameters(), **self.opt_args)
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
    
    def configure_gradient_clipping(self, *args, **kwargs) -> None:
        self.log("train/grad_l2", th.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip))
 
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
        