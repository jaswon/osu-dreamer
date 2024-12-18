
from typing import Any, Callable, Union
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
        self.chunk_size = chunk_size = 1 << args.depth

        # training params
        self.step_ref = step_ref
        self.opt_args = opt_args
        self.cursor_factor = cursor_factor

        # model

        class ChunkPad(nn.Module):
            def __init__(self, pad_value: float = 0.):
                super().__init__()
                self.pad_value = pad_value
                self.chunk_size = chunk_size
        
            def forward(self, x: Float[Tensor, "B D iL"]) -> Float[Tensor, "B D oL"]:
                pad = (self.chunk_size - x.size(-1)%self.chunk_size) % self.chunk_size
                if pad > 0:
                    x = F.pad(x, (0,pad), value=self.pad_value)
                return x

        self.audio_encoder = nn.Sequential(
            ChunkPad(pad_value=-1.),
            MP.Conv1d(A_DIM, args.h_dim, 1),
            MP.PixelNorm(),
            Encoder(args.h_dim, args.depth, args.blocks_per_depth),
            MP.Conv1d(args.h_dim, args.a_dim, 1),
        )
        self.chart_encoder = nn.Sequential(
            ChunkPad(),
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

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        chart: Float[Tensor, str(f"B {X_DIM} L")],
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        za = self.audio_encoder(audio)
        q_zx, vq_loss = self.vq(self.chart_encoder(chart))
        pred_chart = self.decoder(MP.cat([q_zx, za], dim=1))[:,:,:chart.size(-1)]
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

        with th.no_grad():
            za = self.audio_encoder(a)
            q_zx = self.vq(self.chart_encoder(x))[0]
            x_hat = self.decoder(MP.cat([q_zx, za], dim=1))[:,:,:x.size(-1)]
            plots = [ x[0].cpu().numpy() for x in [ x, x_hat, repeat(q_zx, 'b d l -> b d (l r)', r=self.chunk_size)[:,:,:x.size(-1)] ] ]

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
    
    def generate(
        self,
        audio: Float[Tensor, str(f"{A_DIM} L")],
        make_latent: Callable[[Float[Tensor, "A zL"]], Float[Tensor, "B X zL"]],
        return_latents: bool = False,
    ) -> Union[
        Float[Tensor, str(f"B {X_DIM} L")],
        tuple[
            Float[Tensor, str(f"B {X_DIM} L")],
            Float[Tensor, "B A zL"],
            Float[Tensor, "B X zL"],
        ]
    ]:
        za = self.audio_encoder(audio[None])[0]
        pred_zx = make_latent(za)
        q_zx, _ = self.vq(pred_zx)
        za = repeat(za, 'a l -> b a l', b=q_zx.size(0))
        pred_chart = self.decoder(MP.cat([q_zx, za], dim=1))[:,:,:audio.size(-1)].clamp(min=-1, max=1)
        if return_latents:
            return pred_chart, za, pred_zx
        return pred_chart