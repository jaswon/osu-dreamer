
from typing import Any
from jaxtyping import Float

import torch as th
from torch import Tensor
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from einops import repeat, rearrange

import pytorch_lightning as pl

from osu_dreamer.data.beatmap.encode import NUM_LABELS

from osu_dreamer.common.lr_schedule import LRScheduleArgs, make_lr_schedule

from osu_dreamer.data.modules.latent import LatentBatch

from .model import DiffusionModel, DiffusionModelArgs


def frame_dist_sq(
    a: Float[Tensor, "B E l"],
    b: Float[Tensor, "B E l"],
) -> Float[Tensor, "B"]:
    """squared distance in the per-frame metric: sum over channels, mean over length.

    latents are per-frame RMS-normalized, so this metric makes all distance
    statistics independent of sequence length: E[d^2(x0, x1)] = 2E exactly.
    """
    return (a-b).square().sum(1).mean(1)
    
class DiffusionTrainer(pl.LightningModule):
    def __init__(
        self,

        # validation parameters
        val_batches: int,

        # training parameters
        opt_args: dict[str, Any],
        schedule_args: LRScheduleArgs,
        osl_weight: float,
        del_weight: float,

        # model hparams
        emb_dim: int,
        a_dim: int,
        style_dim: int,
        diffusion_args: DiffusionModelArgs,
    ):
        super().__init__()
        th.set_float32_matmul_precision('medium')
        self.save_hyperparameters()

        # validation params
        self.val_batches = val_batches
        
        # training params
        self.opt_args = opt_args
        self.lr_schedule = make_lr_schedule(schedule_args)
        self.osl_weight = osl_weight
        self.del_weight = del_weight

        # model
        self.diffusion = DiffusionModel(emb_dim, a_dim, style_dim, diffusion_args)
        self.diffusion_ema = AveragedModel(self.diffusion, multi_avg_fn=get_ema_multi_avg_fn(.99))

    def forward(
        self, 
        model: DiffusionModel,
        h: Float[Tensor, "B A l"], 
        x1: Float[Tensor, "B E l"], 
        s: Float[Tensor, "B S"],
        _labels: Float[Tensor, str(f"B {NUM_LABELS}")],
    ):
        # stratified logit-normal noise (lower gradient variance)
        B = x1.size(0)
        u = (th.randperm(B, device=x1.device) + th.rand(B, device=x1.device)) / B
        t = th.special.ndtri(u.clamp(1e-6, 1-1e-6)).sigmoid().to(x1.dtype)

        x0 = th.randn_like(x1)
        xt = th.lerp(x0,x1,t[:,None,None])
        u_pred, v_pred = model.forward(h, s, xt)

        # distance marching (arXiv:2602.02928): distances in the per-frame metric
        d_sq = frame_dist_sq(xt, x1)                # (1-t)^2 ||x0-x1||^2
        u_target = (d_sq + model.c0).sqrt()

        # one-step loss: inverse-distance-weighted denoising
        denoised = xt - u_pred[:,None,None] * v_pred
        osl = (frame_dist_sq(denoised, x1) / (d_sq + model.c0)).mean()

        # directional eikonal loss: length-neutral direction supervision
        v_target = (xt - x1) / u_target[:,None,None]
        del_ = frame_dist_sq(v_pred, v_target).mean()

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
        opt = th.optim.AdamW(self.parameters(), **self.opt_args)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": th.optim.lr_scheduler.LambdaLR(opt, self.lr_schedule),
                "interval": "step",
            }
        }
        
    def training_step(self, batch: LatentBatch, batch_idx):
        loss, log_dict = self(self.diffusion, *batch)
        self.log_dict({ f"train/{k}": v for k,v in log_dict.items() })
        return loss

    def on_train_batch_end(self, *args, **kwargs):
        self.diffusion_ema.update_parameters(self.diffusion)
 
    def validation_step(self, batch: LatentBatch, batch_idx, *args, **kwargs):
        h,z,s,l = batch
        
        with th.no_grad():
            seg = z.size(-1) // self.val_batches
            bl = self.val_batches * seg
            h = rearrange(h[...,:bl], '1 ... (b l) -> b ... l', b = self.val_batches)
            z = rearrange(z[...,:bl], '1 ... (b l) -> b ... l', b = self.val_batches)
            s = repeat(s, '1 d -> b d', b = self.val_batches)
            l = repeat(l, '1 d -> b d', b = self.val_batches)
            _, log_dict = self(self.diffusion_ema.module, h,z,s,l)
        self.log_dict({ f"val/{k}": v for k,v in log_dict.items() })