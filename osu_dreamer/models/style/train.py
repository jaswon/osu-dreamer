
from typing import Any
from jaxtyping import Float

import torch as th
from torch import Tensor
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

import pytorch_lightning as pl

from osu_dreamer.data.beatmap.encode import NUM_LABELS, X_DIM
from osu_dreamer.data.module import Batch, pad_to_multiple

from osu_dreamer.common.lr_schedule import LRScheduleArgs, make_lr_schedule

from osu_dreamer.models.latent.train import LatentTrainer

from .model import StyleModel, StyleModelArgs


class StyleTrainer(pl.LightningModule):
    def __init__(
        self,

        # training parameters
        opt_args: dict[str, Any],
        schedule_args: LRScheduleArgs,
        label_drop_prob: float,

        # model hparams
        latent_model_ckpt: str,
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
        self.latent = LatentTrainer.load_from_checkpoint(latent_model_ckpt).latent
        self.latent.requires_grad_(False)
        self.latent.eval()

        self.style = StyleModel(self.latent.style_dim, style_args)

    def on_train_epoch_start(self):
        self.latent.eval()

    def forward(
        self, 
        chart: Float[Tensor, str(f"B {X_DIM} L")], 
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
    ):
        masked_labels = th.where(th.rand_like(labels) < self.label_drop_prob, 0, labels) # classifier free guidance

        with th.no_grad():
            _, s1 = self.latent.encode_chart(chart)

        # minibatch-OT-coupled style noise
        s0 = th.randn_like(s1)
        if s1.size(0) > 1:
            cost = th.cdist(s1.float(), s0.float()).cpu().numpy()
            _, cols = linear_sum_assignment(cost)
            s0 = s0[cols]
        
        t = th.randn(s1.size(0), device=s1.device, dtype=s1.dtype).sigmoid() # logit-normal
        st = th.lerp(s0, s1, t[:,None])
        pred_style_flow = self.style(st, masked_labels, t)
        loss = F.mse_loss(pred_style_flow, s1 - s0)

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

    def on_after_batch_transfer(self, batch: Batch, dataloader_idx: int) -> Batch:
        # pad to chunk_size
        c = self.latent.chunk_size
        audio, chart, labels = batch
        return Batch(pad_to_multiple(audio, c), pad_to_multiple(chart, c), labels)

    def training_step(self, batch: Batch, batch_idx):
        _, chart, labels = batch
        loss, log_dict = self(chart, labels)
        self.log_dict({ f"train/{k}": v for k,v in log_dict.items() })
        return loss

    def validation_step(self, batch: Batch, batch_idx, *args, **kwargs):
        _, chart, labels = batch
        _, log_dict = self(chart, labels)
        self.log_dict({ f"val/{k}": v for k,v in log_dict.items() })
