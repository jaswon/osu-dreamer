
from typing import Any
from jaxtyping import Float

import torch as th
from torch import Tensor
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

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
        osl_weight: float,
        del_weight: float,

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
        self.osl_weight = osl_weight
        self.del_weight = del_weight

        # model
        self.style = StyleModel(style_dim, style_args)
        self.style_ema = AveragedModel(self.style, multi_avg_fn=get_ema_multi_avg_fn(.99))

    def forward(
        self, 
        model: StyleModel,
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
        st = th.lerp(s0, s1, t[:,None])

        masked_labels = th.where(th.rand_like(labels) < self.label_drop_prob, -1, labels)
        u_pred, v_pred = model(st, masked_labels)

        # distance marching (arXiv:2602.02928)
        eps = c0 = model.c0
        d_sq = (st - s1).square().sum(1)
        u_target = (d_sq + c0).sqrt()

        # one-step loss: inverse-distance-weighted denoising
        denoised = st - u_pred[:,None] * v_pred
        osl = ((denoised - s1).square().sum(1) / (d_sq + eps)).mean()

        # directional eikonal loss: length-neutral direction supervision
        v_target = (st - s1) / u_target[:,None]
        del_ = (v_pred - v_target).square().sum(1).mean()

        loss = self.osl_weight * osl + self.del_weight * del_

        # distance estimation error (monitoring only): u vs sqrt(d^2 + c0)
        u_err = ((u_pred - u_target) / u_target).abs().mean()

        return loss, {
            "loss": loss.detach(),
            "osl": osl.detach(),
            "del": del_.detach(),
            "u_mape": u_err.detach(),
        }

    def configure_optimizers(self):
        opt = th.optim.AdamW(self.style.parameters(), **self.opt_args)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": th.optim.lr_scheduler.LambdaLR(opt, self.lr_schedule),
                "interval": "step",
            }
        }

    def training_step(self, batch: LatentBatch, batch_idx):
        loss, log_dict = self(self.style, *batch)
        self.log_dict({ f"train/{k}": v for k,v in log_dict.items() })
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        self.style_ema.update_parameters(self.style)

    def validation_step(self, batch: LatentBatch, batch_idx, *args, **kwargs):
        _, _, s, labels = batch
        self._val_s.append(s.detach())
        self._val_labels.append(labels.detach())

    def on_validation_epoch_start(self):
        self._val_s: list[Tensor] = []
        self._val_labels: list[Tensor] = []

    def on_validation_epoch_end(self):
        s_real = th.cat(self._val_s)      # B S
        labels = th.cat(self._val_labels) # B N
        B = s_real.size(0)
        _, log_dict = self(self.style_ema, th.empty(B,0,0), th.empty(B,0,0), s_real, labels)
        self.log_dict({ f"val/{k}": v for k,v in log_dict.items() })
        if B < 2:
            return
        K = 4
        ema: StyleModel = self.style_ema.module # type: ignore
        samp = th.stack([ema.sample(labels, 16) for _ in range(K)]) # K B S

        d_rr = th.cdist(s_real, s_real).fill_diagonal_(th.inf)
        rr = d_rr.min(1).values.mean()
        flat = samp.flatten(0,1)
        self.log('val/nn_ratio', th.cdist(flat, s_real).min(1).values.mean() / rr)

        hi = labels[:, 0] >= 5
        if hi.sum() > 1:
            R = s_real[hi]
            rr_hi = th.cdist(R, R).fill_diagonal_(th.inf).min(1).values.mean()
            self.log('val/nn_ratio_sr5', th.cdist(samp[:, hi].flatten(0,1), R).min(1).values.mean() / rr_hi)

        # per-condition coverage: closest of the K samples to the true style
        self.log('val/cond_recall', (samp - s_real[None]).norm(dim=-1).min(0).values.mean())

        # sharpness: spread among same-condition samples, relative to real NN spacing
        per_cond = samp.transpose(0,1) # B K S
        self.log('val/sample_spread', th.cdist(per_cond, per_cond).sum() / (K*(K-1)*per_cond.size(0)) / rr)

        self.log('val/energy_dist', energy_distance(flat, s_real))


def energy_distance(x: Float[Tensor, "X D"], y: Float[Tensor, "Y D"]) -> Tensor:
    def mean_dist(a, b, exclude_diag: bool):
        d = th.cdist(a, b)
        if exclude_diag:
            n = a.size(0)
            return d.sum() / (n * (n - 1))
        return d.mean()
    return 2 * mean_dist(x, y, False) - mean_dist(x, x, True) - mean_dist(y, y, True)
