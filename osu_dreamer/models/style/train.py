
from typing import Any
from jaxtyping import Float

import torch as th
from torch import Tensor
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

import pytorch_lightning as pl

from osu_dreamer.data.beatmap.encode import NUM_LABELS
from osu_dreamer.data.modules.latent import LatentBatch

from osu_dreamer.common.lr_schedule import LRScheduleArgs, make_lr_schedule

from .model import StyleModel, StyleModelArgs

class StyleTrainer(pl.LightningModule):
    def __init__(
        self,

        # training parameters
        opt_args: dict[str, Any],
        schedule_args: LRScheduleArgs,
        label_drop_prob: float,

        # model hparams
        style_dim: int,
        style_args: StyleModelArgs,
    ):
        super().__init__()
        th.set_float32_matmul_precision('medium')
        self.save_hyperparameters()

        # training params
        self.opt_args = opt_args
        self.lr_schedule = make_lr_schedule(schedule_args)
        self.label_drop_prob = label_drop_prob

        # model
        self.style = StyleModel(style_dim, style_args)

    def forward(
        self, 
        _h: Float[Tensor, "B A l"], 
        _z: Float[Tensor, "B E l"], 
        s1: Float[Tensor, "B S"],
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
    ):
        B = s1.size(0)

        # stratified logit-normal noise (lower gradient variance)
        u = (th.randperm(B, device=s1.device) + th.rand(B, device=s1.device)) / B
        t = th.special.ndtri(u.clamp(1e-6, 1-1e-6)).sigmoid().to(s1.dtype)

        s0 = th.randn_like(s1)
        if B > 1:
            # minibatch OT-coupled style noise
            cost = th.cdist(s1.float(), s0.float()).cpu().numpy()
            _, cols = linear_sum_assignment(cost)
            s0 = s0[cols]
        st = th.lerp(s0, s1, t[:,None])

        masked_labels = th.where(th.rand_like(labels) < self.label_drop_prob, -1, labels)
        pred_style_flow = self.style(st, masked_labels, t)
        loss = F.mse_loss(pred_style_flow, s1 - s0)
        return loss, {
            "loss": loss.detach(),
        }

    def configure_optimizers(self):
        opt = th.optim.AdamW(self.parameters(), **self.opt_args)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": th.optim.lr_scheduler.LambdaLR(opt, self.lr_schedule),
                "interval": "step",
            }
        }

    def training_step(self, batch: LatentBatch, batch_idx):
        loss, log_dict = self(*batch)
        self.log_dict({ f"train/{k}": v for k,v in log_dict.items() })
        return loss

    def validation_step(self, batch: LatentBatch, batch_idx, *args, **kwargs):
        _, log_dict = self(*batch)
        self.log_dict({ f"val/{k}": v for k,v in log_dict.items() })
