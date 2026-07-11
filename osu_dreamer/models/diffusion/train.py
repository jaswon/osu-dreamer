
from typing import Any
from jaxtyping import Float

import torch as th
from torch import Tensor
import torch.nn.functional as F

from einops import repeat, rearrange

import pytorch_lightning as pl

from osu_dreamer.data.beatmap.encode import NUM_LABELS

from osu_dreamer.common.lr_schedule import LRScheduleArgs, make_lr_schedule

from osu_dreamer.data.modules.latent import LatentBatch

from .model import DiffusionModel, DiffusionModelArgs

    
class DiffusionTrainer(pl.LightningModule):
    def __init__(
        self,

        # validation parameters
        val_batches: int,

        # training parameters
        opt_args: dict[str, Any],
        schedule_args: LRScheduleArgs,
        label_drop_prob: float,

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
        self.label_drop_prob = label_drop_prob

        # model
        self.diffusion = DiffusionModel(emb_dim, a_dim, style_dim, diffusion_args)

    def forward(
        self, 
        h: Float[Tensor, "B A l"], 
        z: Float[Tensor, "B E l"], 
        s: Float[Tensor, "B S"],
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
    ):
        masked_labels = th.where(th.rand_like(labels) < self.label_drop_prob, 0, labels) # classifier free guidance

        # stratified logit-normal noise (lower gradient variance)
        B = z.size(0)
        u = (th.randperm(B, device=z.device) + th.rand(B, device=z.device)) / B
        t = th.special.ndtri(u.clamp(1e-6, 1-1e-6)).sigmoid().to(z.dtype)

        x1 = z
        x0 = th.randn_like(x1)
        true_flow = x1 - x0
        xt = th.lerp(x0,x1,t[:,None,None])

        pred_flow = self.diffusion.forward(h, masked_labels, s, xt, t)
        loss = F.mse_loss(pred_flow, true_flow, reduction='none').sum(dim=1).mean()

        return loss, {
            "loss": loss.detach(),
        }

    def configure_optimizers(self):
        params = [ p for p in self.parameters() if p.requires_grad ]
        opt = th.optim.AdamW(params, **self.opt_args)
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
        h,z,s,l = batch
        
        with th.no_grad():
            seg = z.size(-1) // self.val_batches
            bl = self.val_batches * seg
            h = rearrange(h[...,:bl], '1 ... (b l) -> b ... l', b = self.val_batches)
            z = rearrange(z[...,:bl], '1 ... (b l) -> b ... l', b = self.val_batches)
            s = repeat(s, '1 d -> b d', b = self.val_batches)
            l = repeat(l, '1 d -> b d', b = self.val_batches)
            _, log_dict = self(h,z,s,l)
        self.log_dict({ f"val/{k}": v for k,v in log_dict.items() })