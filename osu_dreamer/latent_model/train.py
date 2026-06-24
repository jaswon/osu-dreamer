
from typing import Any

import torch as th
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.utils.tensorboard.writer import SummaryWriter

from osu_dreamer.data.beatmap.encode import BeatmapEncoding, HitSignals, CursorSignals
from osu_dreamer.data.module import Batch
from osu_dreamer.data.plot import plot_signals

from osu_dreamer.modules.lr_schedule import LRScheduleArgs, make_lr_schedule

from .model import LatentModel, LatentModelArgs

LOSS_COMPONENTS = (
    "hit/onset",
    "hit/combo",
    "hit/slide",
    "hit/sustain",
    "hit/whistle",
    "hit/finish",
    "hit/clap",
    "cursor/pos",
    "cursor/vel",
    "cursor/acc",
    "label",
)

class LatentTrainer(pl.LightningModule):
    def __init__(
        self,

        # training parameters
        opt_args: dict[str, Any],
        schedule_args: LRScheduleArgs,
        kl_weight: float,

        # model hparams
        emb_dim: int,
        n_downs: int,
        stride: int,
        latent_args: LatentModelArgs,
    ):
        super().__init__()
        th.set_float32_matmul_precision('medium')
        self.save_hyperparameters()

        # training params
        self.opt_args = opt_args
        self.lr_schedule = make_lr_schedule(schedule_args)
        self.kl_weight = kl_weight

        self.latent = LatentModel(emb_dim, n_downs, stride, latent_args)
    
    def forward(self, batch: Batch):

        audio, true_chart, true_labels = batch
        pred_chart_logits, pred_labels, kl_loss = self.latent(audio, true_chart)

        hit_loss = F.binary_cross_entropy_with_logits(
            pred_chart_logits[:,HitSignals],
            true_chart[:,HitSignals],
            reduction='none',
        ).mul(1 + 9 * true_chart[:,HitSignals]).mean(dim=(0,2))

        hitting = th.max(true_chart[:, [
            BeatmapEncoding.ONSET,
            BeatmapEncoding.SUSTAIN,
        ]], dim=1, keepdim=True).values
        hitting_w = 1 + 9 * hitting

        cursor_diff_factor = 10

        pos_loss = cursor_diff_factor ** 0 * F.mse_loss(
            pred_chart_logits[:,CursorSignals],
            true_chart[:,CursorSignals],
            reduction='none',
        ).mul(hitting_w).mean()

        vel_loss = cursor_diff_factor ** 1 * F.mse_loss(
            pred_chart_logits[:,CursorSignals].diff(n=1),
            true_chart[:,CursorSignals].diff(n=1),
            reduction='none',
        ).mul(hitting_w[:,:,:-1]).mean()

        acc_loss = cursor_diff_factor ** 2 * F.mse_loss(
            pred_chart_logits[:,CursorSignals].diff(n=2),
            true_chart[:,CursorSignals].diff(n=2),
            reduction='none',
        ).mul(hitting_w[:,:,1:-1]).mean()

        label_loss = F.mse_loss(pred_labels, true_labels, reduction='none').mean()

        losses = th.stack([ *hit_loss.unbind(), pos_loss, vel_loss, acc_loss, label_loss ])
        loss = losses.sum() + self.kl_weight * kl_loss
        return loss, {
            **{ name: loss.detach() for name, loss in zip(LOSS_COMPONENTS, losses) },
            "kl": kl_loss.detach(),
            "loss": loss.detach(),
        }

    def configure_optimizers(self):
        opt = th.optim.AdamW(self.parameters(), **self.opt_args)
        return [opt], [
            {
                "scheduler": th.optim.lr_scheduler.LambdaLR(opt, self.lr_schedule),
                "interval": "step",
            }
        ]

    def training_step(self, batch: Batch, batch_idx):
        loss, log_dict = self(batch)
        self.log_dict({ f"train/{k}": v for k,v in log_dict.items() })
        return loss

    @th.no_grad
    def validation_step(self, batch: Batch, batch_idx, *args, **kwargs):
        if batch_idx == 0:
            self.plot_val(batch)

        _, log_dict = self(batch)
        self.log_dict({ f"val/{k}": v for k,v in log_dict.items() })

    @th.no_grad
    def plot_val(self, b: Batch):
        from einops import repeat

        a,x,_ = b
        z = self.latent.encode(x)
        plot_z = repeat(z, 'b d l -> b d (l r)', r=self.latent.chunk_size)[:,:,:x.size(-1)]
        pred_x, _ = self.latent.decode(a, z)

        exp: SummaryWriter = self.logger.experiment # type: ignore
        with plot_signals(
            a[0].cpu().numpy(),
            [ s[0].cpu().float().numpy() for s in [ x, pred_x, x-pred_x, plot_z ] ],
        ) as fig:
            exp.add_figure("samples", fig, global_step=self.global_step)