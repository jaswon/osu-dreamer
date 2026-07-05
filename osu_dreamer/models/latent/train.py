
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
        z_reg_weight: float,
        z_dropout: float,

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
        self.z_reg_weight = z_reg_weight
        self.z_dropout = z_dropout

        self.latent = LatentModel(emb_dim, n_downs, stride, latent_args)
    
    def forward(self, batch: Batch):

        audio, true_chart, true_labels = batch
        
        mu, logvar = self.latent.param_encode(true_chart)
        z_reg_loss = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(dim=1).mean()
        z = mu + th.exp(0.5 * logvar) * th.randn_like(mu)
        z[th.rand(z.size(0)) < self.z_dropout] = 0

        pred_chart_logits, pred_labels = self.latent(audio, z)

        hit_loss = F.binary_cross_entropy_with_logits(
            pred_chart_logits[:,HitSignals],
            true_chart[:,HitSignals],
            reduction='none',
        ).mul(1 + 9 * true_chart[:,HitSignals]).mean(dim=(0,2))

        cursor_losses = [
            F.mse_loss(
                pred_chart_logits[:,CursorSignals].diff(n=i),
                true_chart[:,CursorSignals].diff(n=i),
            ) * 10 ** (i+1)
            for i in range(3)
        ]

        label_loss = F.mse_loss(pred_labels, true_labels)

        losses = th.stack([ *hit_loss.unbind(), *cursor_losses, label_loss ])
        loss = (
            losses.sum()
            + self.z_reg_weight * z_reg_loss
        )
        return loss, {
            **{ name: loss.detach() for name, loss in zip(LOSS_COMPONENTS, losses) },
            "z_reg": z_reg_loss.detach(),
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

    def on_validation_epoch_start(self):
        # onset soft-Dice sums (shared true-norm `_on_tt`); `_sh` = shuffled latent
        self._on_pt = self._on_pp = self._on_tt = 0.
        self._on_pt_sh = self._on_pp_sh = 0.
        # cursor R² sums: residual and total (variance) sum-of-squares
        self._cur_res = self._cur_tot = 0.

    @th.no_grad
    def validation_step(self, batch: Batch, batch_idx, *args, **kwargs):
        if batch_idx == 0:
            self.plot_val(batch)

        _, log_dict = self(batch)
        self.log_dict({ f"val/{k}": v for k,v in log_dict.items() })
        self.log_dict(self.eval_metrics(batch))

    def on_validation_epoch_end(self):
        def soft_f1(pt, pp, tt):
            # 2<p,t> / (<p,p> + <t,t>): threshold-free Dice / soft F1 in [0,1]
            return 2 * pt / max(pp + tt, 1e-8)

        def hmean(a, b):
            # harmonic mean: near-zero if either term is neglected
            return 2 * a * b / max(a + b, 1e-8)

        onset_f1 = soft_f1(self._on_pt, self._on_pp, self._on_tt)
        onset_f1_sh = soft_f1(self._on_pt_sh, self._on_pp_sh, self._on_tt)
        cursor_r2 = 1. - self._cur_res / max(self._cur_tot, 1e-8)
        cursor_q = self._cur_tot / max(self._cur_tot + self._cur_res, 1e-8)

        self.log_dict({
            "eval/onset_soft_f1": onset_f1,
            "eval/cursor_vel_r2": cursor_r2,
            "eval/score": hmean(onset_f1, cursor_q),
            # onset overlap with a temporally-shuffled (mismatched but in-
            # distribution) latent; gain ≈ 0 => decoder recovers onsets from
            # audio, gain ≈ onset_f1 => onsets are driven by the latent
            "eval/onset_soft_f1_shuf": onset_f1_sh,
            "eval/onset_soft_f1_gain": onset_f1 - onset_f1_sh,
        })

    @th.no_grad
    def eval_metrics(self, b: Batch) -> dict[str, th.Tensor]:
        """
        interpretable, sampling-free reconstruction metrics.
        assumes the val dataloader uses batch_size=1 (full-length maps).
        """
        a, x, true_labels = b

        h = self.latent.encoder(x)
        mu, logvar = self.latent.mu(h), self.latent.logvar(h)

        pred_chart, pred_labels = self.latent.decode(a, mu)
        # shuffled latent: a real, in-distribution z that no longer matches the
        # audio (temporal roll). unlike zeroing z, this avoids OOD artifacts, so
        # it cleanly isolates whether the decoder relies on the latent vs audio
        z_shuf = th.roll(mu, shifts=max(1, mu.size(-1) // 2), dims=-1)
        pred_chart_sh, _ = self.latent.decode(a, z_shuf)

        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1. - logvar).mean(dim=(0, 2))
        active_units = (kl_per_dim > 1e-2).sum().float()

        # --- onset soft-F1 (Dice) accumulation ---
        # continuous overlap of the onset activation curves; equals F1 for binary
        # signals but needs no peak-picking or timing tolerance
        t = x[:, BeatmapEncoding.ONSET].float()
        p = pred_chart[:, BeatmapEncoding.ONSET].float()
        p_sh = pred_chart_sh[:, BeatmapEncoding.ONSET].float()
        self._on_tt += t.mul(t).sum().item()
        self._on_pt += p.mul(t).sum().item()
        self._on_pp += p.mul(p).sum().item()
        self._on_pt_sh += p_sh.mul(t).sum().item()
        self._on_pp_sh += p_sh.mul(p_sh).sum().item()

        # --- cursor R² accumulation ---
        # computed on velocity (first difference), not raw position: position R²
        # saturates near 1 because the low-frequency gross trajectory dominates
        # total variance; velocity isolates the fine motion that actually differs
        scale = x.new_tensor([512., 384.]).float()[None, :, None]
        true_xy = x[:, CursorSignals].float() * scale
        pred_xy = pred_chart[:, CursorSignals].float() * scale
        true_v = true_xy.diff(dim=-1)
        pred_v = pred_xy.diff(dim=-1)
        self._cur_res += (pred_v - true_v).pow(2).sum().item()
        self._cur_tot += (true_v - true_v.mean(dim=-1, keepdim=True)).pow(2).sum().item()

        # interpretable diagnostics (unit conversion only, not thresholds)
        cursor_px = (pred_xy - true_xy).abs().mean()
        cursor_px_sh = ((pred_chart_sh[:, CursorSignals].float() * scale) - true_xy).abs().mean()
        label_mae = (pred_labels - true_labels).abs().mean()

        return {
            "eval/cursor_px_mae": cursor_px,
            "eval/label_mae": label_mae,
            "eval/active_units": active_units,
            "eval/kl_mean": kl_per_dim.mean(),
            "eval/cursor_px_ablation": cursor_px_sh - cursor_px,
        }

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