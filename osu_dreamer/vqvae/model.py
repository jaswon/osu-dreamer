
from typing import Any
from jaxtyping import Float, Int

from dataclasses import dataclass

import numpy as np

import torch as th
from torch import Tensor, nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.utils.tensorboard.writer import SummaryWriter

from osu_dreamer.data.beatmap.encode import X_DIM
from osu_dreamer.data.plot import plot_signals

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.muon import Muon

from .data.module import Batch

from .modules.hard_attn import HardAttn, HardAttnArgs
from .modules.ae import Encoder, Decoder, AutoEncoderArgs
from .modules.critic import Critic, CriticArgs

@dataclass
class PriorFactorScheduleArgs:
    midpoint: int
    rate: float
    max: float = 1.

def make_prior_factor_schedule(args: PriorFactorScheduleArgs):
    def get_factor(step: int) -> float:
        return args.max / (1. + np.exp(-args.rate * (step - args.midpoint)))
    return get_factor

    
class Model(pl.LightningModule):
    def __init__(
        self,

        # training parameters
        opt_args: dict[str, Any],
        grad_clip: float,
        gp_factor: float,
        gan_factor: float,
        prior_schedule: PriorFactorScheduleArgs,

        # model hparams
        emb_dim: int,
        ae_args: AutoEncoderArgs,
        hard_attn_args: HardAttnArgs,
        critics: list[CriticArgs],
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.chunk_size = ae_args.stride ** ae_args.depth
        self.emb_dim = emb_dim

        # training params
        self.opt_args = opt_args
        self.grad_clip = grad_clip
        self.gp_factor = gp_factor
        self.gan_factor = gan_factor
        self.prior_schedule = make_prior_factor_schedule(prior_schedule)

        # model
        self.hard_attn = HardAttn(emb_dim, hard_attn_args)
        self.encoder = nn.Sequential(
            MP.Conv1d(X_DIM, emb_dim, 1),
            Encoder(emb_dim, ae_args),
        )
        self.decoder = nn.Sequential(
            Decoder(emb_dim, ae_args),
            MP.Conv1d(emb_dim, X_DIM, 1),
            MP.Gain(),
        )
        self.critics = nn.ModuleList([
            Critic(X_DIM, conf)
            for conf in critics
        ])

    def padding(self, L: int) -> int:
        """returns the amount of padding required to align a sequence of length L"""
        a = self.chunk_size
        return (a-L%a)%a

    def forward(
        self,
        true_chart: Float[Tensor, str(f"B {X_DIM} L")],
    ) -> tuple[
        Float[Tensor, str(f"B {X_DIM} L")],
        Float[Tensor, ""],
        Int[Tensor, "B H l"],
    ]:
        z = self.encoder(true_chart)
        z_q, entropy, indices = self.hard_attn(z)
        return self.decoder(z_q), entropy, indices
    
    @th.no_grad
    def encode(
        self,
        chart: Float[Tensor, str(f"B {X_DIM} L")],
    ) -> Float[Tensor, "B D l"]:
        pad = self.padding(chart.size(-1))
        if pad > 0:
            chart = F.pad(chart, (0, pad))
        z = self.encoder(chart) # B H l
        z_q, _, _ = self.hard_attn(z)
        return z_q
    
    @th.no_grad
    def decode(
        self,
        z_q: Float[Tensor, "B D l"]
    ) -> Float[Tensor, str(f"B {X_DIM} L")]:
        return self.decoder(z_q)

#
#
# =============================================================================
# MODEL TRAINING
# =============================================================================
#
#

    def configure_optimizers(self):
        c_opt = Muon(self.critics.parameters(), **self.opt_args)
        g_opt = Muon([
            *self.hard_attn.parameters(),
            *self.encoder.parameters(),
            *self.decoder.parameters(),
        ], **self.opt_args)
        return c_opt, g_opt

    def training_step(self, batch: Batch, batch_idx):
        opts: list[LightningOptimizer] = self.optimizers() # type: ignore
        c_opt, g_opt = opts
        _, true_chart = batch
        B = true_chart.size(0)

        pad = self.padding(true_chart.size(-1))
        if pad > 0:
            true_chart = F.pad(true_chart, (0,pad))

        # train critics
        pred_chart = self(true_chart)[0].detach()

        critic_adv_loss = th.tensor(0, device=self.device)
        gradient_penalty = th.tensor(0, device=self.device)
        for critic in self.critics:
            *_, pred_score = critic(pred_chart)
            *_, true_score = critic(true_chart)

            critic_adv_loss = critic_adv_loss + ( -true_score.mean() + pred_score.mean() )

            # gradient penalty
            alpha = th.rand(B,1,1, device=self.device)
            interpolates = ( alpha * true_chart + (1-alpha) * pred_chart ).requires_grad_(True)
            *_, interpolates_score = critic(interpolates)
            gradients = th.autograd.grad(
                outputs=interpolates_score,
                inputs=interpolates,
                grad_outputs=th.ones_like(interpolates_score),
                create_graph=True,
                retain_graph=True,
            )[0].view(B, -1)
            gradient_penalty = gradient_penalty + ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        critic_loss = critic_adv_loss + self.gp_factor * gradient_penalty
        c_opt.zero_grad()
        self.manual_backward(critic_loss)
        c_opt.step()
        c_opt.zero_grad()
        self.log_dict({
            "train/critic/adversarial": critic_adv_loss.detach(),
            "train/critic/gradient_penalty": gradient_penalty.detach(),
        })

        # train generator
        pred_chart, pred_entropy, pred_indices = self(true_chart)
        perplexity = self.hard_attn.compute_perplexity(pred_indices)
        prior_loss = -pred_entropy
        with th.no_grad():
            l2_loss = (pred_chart - true_chart).pow(2).mean()

        gen_adv_loss = th.tensor(0, device=self.device)
        rec_loss = th.tensor(0, device=self.device)
        for critic in self.critics:
            *pred_fmaps, pred_score = critic(pred_chart)
            *true_fmaps, _          = critic(true_chart)

            gen_adv_loss = gen_adv_loss + ( -pred_score.mean() )

            # feature matching
            for pred_fmap, true_fmap in zip(pred_fmaps, true_fmaps):
                rec_loss_i = (pred_fmap - true_fmap).abs().mean() / pred_fmap.size(1)
                rec_loss = rec_loss + rec_loss_i

        prior_factor = self.prior_schedule(self.global_step)
        gen_loss = rec_loss + self.gan_factor * gen_adv_loss + prior_factor * prior_loss
        g_opt.zero_grad()
        self.manual_backward(gen_loss)
        if self.grad_clip > 0:
            g_opt_params = [ p for g in g_opt.param_groups for p in g['params'] if p.grad is not None ]
            self.log("train/gen/grad_l2", th.nn.utils.clip_grad_norm_(g_opt_params, self.grad_clip).item())
        g_opt.step()
        g_opt.zero_grad()
        self.log_dict({
            "train/gen/prior_factor": prior_factor,
            "train/gen/prior": prior_loss.detach(),
            "train/gen/adversarial": gen_adv_loss.detach(),
            "train/gen/reconstruction": rec_loss.detach(),
            "train/gen/perplexity": perplexity,
            "train/gen/l2": l2_loss,
        })

    @th.no_grad
    def validation_step(self, batch: Batch, batch_idx, *args, **kwargs):
        if batch_idx == 0:
            self.plot_val(batch)

        _, true_chart = batch

        pad = self.padding(true_chart.size(-1))
        if pad > 0:
            true_chart = F.pad(true_chart, (0,pad))

        pred_chart, _, _ = self(true_chart)
        self.log_dict({
            'val/l2_rec': (true_chart - pred_chart).pow(2).mean(),
        })

    @th.no_grad
    def plot_val(self, b: Batch):
        a,x = b
        pred_x = self.decode(self.encode(x))[:,:,:x.size(-1)]

        exp: SummaryWriter = self.logger.experiment # type: ignore
        with plot_signals(
            a[0].cpu().numpy(),
            [ x[0].cpu().numpy() for x in [ x, pred_x ] ],
        ) as fig:
            exp.add_figure("samples", fig, global_step=self.global_step)
        