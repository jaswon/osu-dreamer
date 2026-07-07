
from typing import Any
from jaxtyping import Float

import torch as th
from torch import Tensor
import torch.nn.functional as F

from einops import repeat, rearrange

from scipy.optimize import linear_sum_assignment

import pytorch_lightning as pl
from torch.utils.tensorboard.writer import SummaryWriter

from osu_dreamer.data.beatmap.encode import NUM_LABELS, X_DIM
from osu_dreamer.data.load_audio import A_DIM
from osu_dreamer.data.module import Batch, pad_to_multiple
from osu_dreamer.data.plot import plot_signals

from osu_dreamer.modules.lr_schedule import LRScheduleArgs, make_lr_schedule
from osu_dreamer.modules.wae import mmd_imq

from osu_dreamer.models.latent.train import LatentTrainer

from .model import DiffusionModel, DiffusionModelArgs
from .posterior import FlowPosterior, FlowPosteriorArgs

    
class DiffusionTrainer(pl.LightningModule):
    def __init__(
        self,

        # validation parameters
        val_batches: int,
        val_steps: int,

        # training parameters
        opt_args: dict[str, Any],
        schedule_args: LRScheduleArgs,
        label_drop_prob: float,
        flow_reg_factor: float,

        # model hparams
        latent_model_ckpt: str,
        flow_latent_dim: int,
        posterior_args: FlowPosteriorArgs,
        diffusion_args: DiffusionModelArgs,
    ):
        super().__init__()
        th.set_float32_matmul_precision('medium')
        self.save_hyperparameters()

        # validation params
        self.val_batches = val_batches
        self.val_steps = val_steps

        # training params
        self.opt_args = opt_args
        self.lr_schedule = make_lr_schedule(schedule_args)
        self.label_drop_prob = label_drop_prob
        self.flow_reg_factor = flow_reg_factor

        # model
        self.latent = LatentTrainer.load_from_checkpoint(latent_model_ckpt).latent
        self.diffusion = DiffusionModel(self.latent.emb_dim, self.latent.a_dim, flow_latent_dim, diffusion_args)
        self.posterior = FlowPosterior(self.latent.emb_dim + self.latent.a_dim, flow_latent_dim, posterior_args)

    def forward(
        self, 
        audio: Float[Tensor, str(f"B {A_DIM} L")], 
        chart: Float[Tensor, str(f"B {X_DIM} L")], 
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
    ):
        masked_labels = th.where(th.rand_like(labels) < self.label_drop_prob, 0, labels) # classifier free guidance
        
        with th.no_grad():
            x1 = self.latent.encode_chart(chart)
            _, a = self.latent.audio_encoder(audio)
        x0 = self.ot_coupled_noise(x1)
        true_flow = x1 - x0
        t = th.randn(audio.size(0), device=x1.device, dtype=x1.dtype).sigmoid() # logit-normal
        xt = th.lerp(x0,x1,t[:,None,None])

        flow_latent = self.posterior(th.cat([a, x1], dim=1))
        flow_reg_loss = mmd_imq(flow_latent, th.randn_like(flow_latent))

        pred_flow = self.diffusion.forward(a, masked_labels, flow_latent, xt, t)
        recon_loss = F.mse_loss(pred_flow, true_flow, reduction='none').sum(dim=1).mean()

        loss = recon_loss + self.flow_reg_factor * flow_reg_loss
        return loss, {
            "loss": loss.detach(),
            "recon": recon_loss.detach(),
            "flow_reg": flow_reg_loss.detach(),
        }

    @th.no_grad()
    def ot_coupled_noise(self, x1: Float[Tensor, "B E L"]) -> Float[Tensor, "B E L"]:
        """sample noise minibatch-OT-coupled to `x1` (straightens flow paths)"""
        x0 = th.randn_like(x1)
        if x1.size(0) < 2:
            return x0
        cost = th.cdist(
            x1.flatten(1).float(),
            x0.flatten(1).float(),
        ).cpu().numpy()
        _, cols = linear_sum_assignment(cost)
        return x0[cols]

    def configure_optimizers(self):
        opt = th.optim.AdamW(self.parameters(), **self.opt_args)
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
        loss, log_dict = self(*batch)
        self.log_dict({ f"train/{k}": v for k,v in log_dict.items() })
        return loss
 
    def validation_step(self, batch: Batch, batch_idx, *args, **kwargs):
        a,x,l = batch

        if batch_idx == 0:
            skips, h = self.latent.audio_encoder(a)
            pred_z = self.diffusion.sample(h, l, self.val_steps)
            pred_x, _ = self.latent.decode(pred_z, skips=skips)

            exp: SummaryWriter = self.logger.experiment # type: ignore
            with plot_signals(
                a[0].cpu().numpy(),
                [ x[0].cpu().float().numpy() for x in [ x, pred_x ] ],
            ) as fig:
                exp.add_figure("samples", fig, global_step=self.global_step)
        
        with th.no_grad():
            c = self.latent.chunk_size
            seg = (a.size(-1) // self.val_batches) // c * c
            bL = self.val_batches * seg
            a = rearrange(a[...,:bL], '1 ... (b l) -> b ... l', b = self.val_batches)
            x = rearrange(x[...,:bL], '1 ... (b l) -> b ... l', b = self.val_batches)
            l = repeat(l, '1 d -> b d', b = self.val_batches)
            _, log_dict = self(a,x,l)
        self.log_dict({ f"val/{k}": v for k,v in log_dict.items() })