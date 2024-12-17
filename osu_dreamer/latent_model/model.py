
from typing import Any, Callable
from jaxtyping import Float

from dataclasses import dataclass

import torch as th
from torch import Tensor, nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.utils.tensorboard.writer import SummaryWriter

from einops import repeat

from osu_dreamer.data.dataset import Batch
from osu_dreamer.data.load_audio import A_DIM
from osu_dreamer.data.beatmap.encode import X_DIM, CursorSignals
from osu_dreamer.data.labels import NUM_LABELS
from osu_dreamer.data.plot import plot_signals

from osu_dreamer.modules.adabelief import AdaBelief
import osu_dreamer.modules.mp as MP

from .modules import Encoder, Decoder, VectorQuantizer

@dataclass
class VQGANArgs:
    x_dim: int
    a_dim: int
    h_dim: int
    depth: int
    blocks_per_depth: int
    vocab_size: int
    
class Model(pl.LightningModule):
    def __init__(
        self,

        # training parameters
        opt_args: dict[str, Any],
        step_ref: float,
        cursor_factor: float,

        # model hparams
        args: VQGANArgs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.x_dim = args.x_dim
        self.a_dim = args.a_dim
        self.chunk_size = 1 << args.depth

        # training params
        self.step_ref = step_ref
        self.opt_args = opt_args
        self.cursor_factor = cursor_factor

        # model

        self.audio_encoder = nn.Sequential(
            MP.Conv1d(A_DIM, args.h_dim, 1),
            MP.PixelNorm(),
            Encoder(args.h_dim, args.depth, args.blocks_per_depth),
            MP.Conv1d(args.h_dim, args.a_dim, 1),
        )
        self.chart_encoder = nn.Sequential(
            MP.Conv1d(X_DIM, args.h_dim, 1),
            MP.PixelNorm(),
            Encoder(args.h_dim, args.depth, args.blocks_per_depth),
            MP.Conv1d(args.h_dim, args.x_dim, 1),
        )
        
        self.vq = VectorQuantizer(args.x_dim, args.vocab_size)
        self.decoder = nn.Sequential(
            MP.Conv1d(args.a_dim + args.x_dim, args.h_dim, 1),
            Decoder(args.h_dim, args.depth, args.blocks_per_depth),
            MP.Conv1d(args.h_dim, X_DIM, 1),
            MP.Gain(),
        )

    def _padding(self, L: int):
        return (self.chunk_size-L%self.chunk_size)%self.chunk_size

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        chart: Float[Tensor, str(f"B {X_DIM} L")],
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        pad = self._padding(audio.size(-1))
        if pad > 0:
            chart = F.pad(chart, (0,pad))
            audio = F.pad(audio, (0,pad), value=-1)

        z = self.chart_encoder(chart)
        q_z, vq_loss = self.vq(z)
        pred_chart = self.decoder(MP.cat([q_z, self.audio_encoder(audio)], dim=1))
        recon_loss = th.mean((chart - pred_chart) ** 2)
        bound_loss = th.mean((pred_chart.abs().clamp(min=1) - 1) ** 2)

        cursor_diff = chart[:, CursorSignals, 1:] - chart[:, CursorSignals, :-1]
        pred_cursor_diff = pred_chart[:, CursorSignals, 1:] - pred_chart[:, CursorSignals, :-1]
        cd_map = lambda diff: th.tanh(diff * 20)
        cursor_loss = (cd_map(cursor_diff) - cd_map(pred_cursor_diff)).pow(2).mean()

        
        loss = recon_loss + cursor_loss * self.cursor_factor + bound_loss + vq_loss
        return loss, {
            'loss': loss.detach(),
            'recon': recon_loss.detach(),
            'cursor': cursor_loss.detach(),
            'bound': bound_loss.detach(),
            'vq': vq_loss.detach(),
        }
        
#
#
# =============================================================================
# MODEL TRAINING
# =============================================================================
#
#

    def configure_optimizers(self):
        opt = AdaBelief(self.parameters(), **self.opt_args)
        isqrt = lambda step: max(step / self.step_ref, 1) ** -.5
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": th.optim.lr_scheduler.LambdaLR(opt, isqrt),
                "interval": "step",
            }
        }

    def training_step(self, batch: Batch, batch_idx):
        loss, log_dict = self.forward(*batch)
        self.log_dict({ f"train/{k}": v for k,v in log_dict.items() })
        return loss
 
    def validation_step(self, batch: Batch, batch_idx, *args, **kwargs):
        with th.no_grad():
            _, log_dict = self.forward(*batch)
        self.log_dict({ f"val/{k}": v for k,v in log_dict.items() })

        if batch_idx == 0:
            self.plot_sample(batch)

    def plot_sample(self, b: Batch):
        a, x, _ = b
        
        padding = (self.chunk_size-x.size(-1)%self.chunk_size)%self.chunk_size
        if padding > 0:
            x = F.pad(x, (0, padding))
            a = F.pad(a, (0,padding), value=-1)

        with th.no_grad():
            za = self.audio_encoder(a)
            q_zx = self.vq(self.chart_encoder(x))[0]
            x_hat = self.decoder(MP.cat([q_zx, za], dim=1))
            plots = [ x[0].cpu().numpy() for x in [ x, x_hat, repeat(q_zx, 'b d l -> b d (l r)', r=self.chunk_size) ] ]

        with plot_signals(a[0].cpu().numpy(), plots) as fig:
            exp: SummaryWriter = self.logger.experiment # type: ignore
            exp.add_figure("samples", fig, global_step=self.global_step)
        
#
#
# =============================================================================
# MODEL API
# =============================================================================
#
#

    def encode(
        self, 
        chart: Float[Tensor, str(f"B {X_DIM} L")],
        audio: Float[Tensor, str(f"B {A_DIM} L")],
    ) -> tuple[
        Float[Tensor, "B Zx zL"],
        Float[Tensor, "B Za zL"],
    ]:
        pad = self._padding(audio.size(-1))
        if pad > 0:
            chart = F.pad(chart, (0,pad))
            audio = F.pad(audio, (0,pad), value=-1)
        z_x = self.chart_encoder(chart)
        z_a = self.audio_encoder(audio)
        return z_x, z_a
    
    def generate(
        self,
        audio: Float[Tensor, str(f"{A_DIM} L")],
        make_latent: Callable[[Float[Tensor, "A zL"]], Float[Tensor, "B X zL"]],
    ) -> Float[Tensor, str(f"B {X_DIM} L")]:
        pad = self._padding(audio.size(-1))
        if pad > 0:
            audio = F.pad(audio, (0,pad), value=-1)
        za = self.audio_encoder(audio[None])[0]
        zx = make_latent(za)
        za = repeat(za, 'a l -> b a l', b=zx.size(0))
        q_zx, _ = self.vq(zx)
        pred_chart = self.decoder(MP.cat([q_zx, za], dim=1))
        if pad > 0:
            pred_chart = pred_chart[...,:-pad]
        return pred_chart.clamp(min=-1, max=1)