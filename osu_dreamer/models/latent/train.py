
from typing import Any

import torch as th
import torch.nn.functional as F

from einops import rearrange, repeat

import pytorch_lightning as pl
from torch.utils.tensorboard.writer import SummaryWriter

from osu_dreamer.data.beatmap.encode import BeatmapEncoding, HitSignals, CursorSignals
from osu_dreamer.data.modules.beatmap import Batch, pad_to_multiple
from osu_dreamer.data.plot import plot_signals

from osu_dreamer.common.lr_schedule import LRScheduleArgs, make_lr_schedule
from osu_dreamer.common.wae import mmd_imq

from .model import LatentModel, LatentModelArgs

LOSS_COMPONENT_WEIGHTS = {
    "hit/onset": 1,
    "hit/combo": 1,
    "hit/slide": 1,
    "hit/sustain": 1,
    "hit/whistle": 1,
    "hit/finish": 1,
    "hit/clap": 1,
    "cursor/pos": 2,
    "cursor/vel": 2,
    "cursor/acc": 2,
    "label": 2,
}

class LatentTrainer(pl.LightningModule):
    def __init__(
        self,

        # training parameters
        opt_args: dict[str, Any],
        schedule_args: LRScheduleArgs,
        s_reg_weight: float,
        s_noise: float,
        z_noise: float,
        s_mask_frac: float,
        z_mask_frac: float,

        # model hparams
        emb_dim: int,
        style_dim: int,
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
        self.s_reg_weight = s_reg_weight
        self.s_noise = s_noise
        self.z_noise = z_noise
        self.z_mask_frac = z_mask_frac
        self.s_mask_frac = s_mask_frac

        self.loss_ema: th.Tensor
        self.register_buffer('loss_ema', th.ones(len(LOSS_COMPONENT_WEIGHTS)))
        self.loss_ema_initialized: th.Tensor
        self.register_buffer('loss_ema_initialized', th.tensor(False))

        self.latent = LatentModel(emb_dim, style_dim, n_downs, stride, latent_args)
    
    def forward(self, batch: Batch):

        audio, true_chart, true_labels = batch

        # split each window into halves treated as separate batch items; each
        # half is decoded with the *other* half's style code, so style
        # consistency is enforced by the reconstruction loss itself
        audio = rearrange(audio, 'b d (h l) -> (b h) d l', h=2)
        true_chart = rearrange(true_chart, 'b d (h l) -> (b h) d l', h=2)
        true_labels = repeat(true_labels, 'b d -> (b h) d', h=2)

        z, s = self.latent.encode_chart(true_chart)

        s_reg_loss = mmd_imq(s, th.randn_like(s))

        # swap styles within each half-pair
        s_pairs = rearrange(s, '(b h) d -> b h d', h=2)
        s = rearrange(s_pairs.flip(1), 'b h d -> (b h) d')

        s_masked = th.zeros(s.shape[0], dtype=th.bool, device=s.device)
        if self.training:
            s = s + self.s_noise * th.randn_like(s)
            z = z + self.z_noise * th.randn_like(z)

            if self.s_mask_frac > 0:
                s_masked = th.rand(s.shape[0], device=s.device) < self.s_mask_frac
                s = th.where(s_masked[:, None], th.randn_like(s), s)
                
            if self.z_mask_frac > 0:
                # zero out a random contiguous span per item: the decoder must
                # fill gaps from `s` + audio skips, so time-invariant info is
                # cheaper to route through `s` than to replicate in `z`
                B, _, l = z.shape
                span = (th.rand(B, device=z.device) * self.z_mask_frac * l).long()
                start = (th.rand(B, device=z.device) * (l - span).clamp(min=1)).long()
                idx = th.arange(l, device=z.device)[None]
                mask = (idx >= start[:, None]) & (idx < (start + span)[:, None])
                z = z.masked_fill(mask[:, None, :], 0.)
        pred_chart_logits, pred_labels = self.latent(audio, z, s)

        true_hits = true_chart[:,HitSignals]
        # `hit_floor``: soft target bce floor > 0 - subtract for cleaner objective (autocast safe)
        hit_floor = -th.special.xlogy(true_hits, true_hits) - th.special.xlogy(1 - true_hits, 1 - true_hits)
        hit_bce = F.binary_cross_entropy_with_logits(
            pred_chart_logits[:,HitSignals],
            true_hits,
            reduction='none',
        ) - hit_floor
        hit_loss = hit_bce.mean(dim=(0,2))

        cursor_losses = [
            F.mse_loss(
                pred_chart_logits[:,CursorSignals].diff(n=i),
                true_chart[:,CursorSignals].diff(n=i),
            )
            for i in range(3)
        ]

        label_sq_err = (pred_labels - true_labels).pow(2).mean(dim=1)
        label_loss = th.where(s_masked, 0., label_sq_err).sum() / (~s_masked).sum().clamp(min=1)

        losses = th.stack([ *hit_loss.unbind(), *cursor_losses, label_loss ])
        loss_weights = losses.new_tensor(list(LOSS_COMPONENT_WEIGHTS.values()))

        if self.training:
            if not self.loss_ema_initialized:
                self.loss_ema.copy_(losses.detach())
                self.loss_ema_initialized.fill_(True)
            else:
                self.loss_ema.lerp_(losses.detach(), 0.01)

        loss = (
            (loss_weights * losses / self.loss_ema.clamp(min=1e-8)).sum()
            + self.s_reg_weight * s_reg_loss
        )
        return loss, {
            **{ name: loss.detach() for name, loss in zip(LOSS_COMPONENT_WEIGHTS.keys(), losses) },
            "s_reg": s_reg_loss.detach(),
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
    
    def on_after_batch_transfer(self, batch: Batch, dataloader_idx: int) -> Batch:
        # pad to chunk_size
        c = 2 * self.latent.chunk_size
        audio, chart, labels = batch
        return Batch(pad_to_multiple(audio, c), pad_to_multiple(chart, c), labels)

    def training_step(self, batch: Batch, batch_idx):
        loss, log_dict = self(batch)
        self.log_dict({ f"train/{k}": v for k,v in log_dict.items() })
        return loss

    def on_validation_epoch_start(self):
        # onset soft-Dice sums (shared true-norm `_on_tt`)
        self._on_pt = self._on_pp = self._on_tt = 0.
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
        cursor_r2 = 1. - self._cur_res / max(self._cur_tot, 1e-8)
        cursor_q = self._cur_tot / max(self._cur_tot + self._cur_res, 1e-8)

        self.log_dict({
            "eval/hit/dice": onset_f1,
            "eval/cursor/vel/r2": cursor_r2,
            "eval/score": hmean(onset_f1, cursor_q),
        })

    @th.no_grad
    def eval_metrics(self, b: Batch) -> dict[str, th.Tensor]:
        """
        interpretable, sampling-free reconstruction metrics.
        assumes the val dataloader uses batch_size=1 (full-length maps).
        """
        a, x, true_labels = b

        z, s = self.latent.encode_chart(x)

        pred_chart, pred_labels = self.latent.decode(z, s, audio=a)

        # RMSNorm pins mean per-dim energy to ~1, so mean variance is
        # uninformative; the *min* per-dim variance still exposes dead dims
        # (posterior collapse under a deterministic WAE encoder)
        z_var_min = z.var(dim=(0, 2)).min()

        # --- onset soft-F1 (Dice) accumulation ---
        # continuous overlap of the onset activation curves; equals F1 for binary
        # signals but needs no peak-picking or timing tolerance
        t = x[:, BeatmapEncoding.ONSET].float()
        p = pred_chart[:, BeatmapEncoding.ONSET].float()
        self._on_tt += t.mul(t).sum().item()
        self._on_pt += p.mul(t).sum().item()
        self._on_pp += p.mul(p).sum().item()

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

        cursor_px = (pred_xy - true_xy).abs().mean()
        label_mae = (pred_labels - true_labels).abs().mean()

        return {
            "eval/cursor_px_mae": cursor_px,
            "eval/label_mae": label_mae,
            "eval/z_var_min": z_var_min,
        }

    @th.no_grad
    def plot_val(self, b: Batch):
        from einops import repeat

        a,x,_ = b
        z, s = self.latent.encode_chart(x)
        plot_z = repeat(z, 'b d l -> b d (l r)', r=self.latent.chunk_size)[:,:,:x.size(-1)]
        pred_x, _ = self.latent.decode(z, s, audio=a)

        exp: SummaryWriter = self.logger.experiment # type: ignore
        with plot_signals(
            a[0].cpu().numpy(),
            [ s[0].cpu().float().numpy() for s in [ x, pred_x, x-pred_x, plot_z ] ],
        ) as fig:
            exp.add_figure("samples", fig, global_step=self.global_step)