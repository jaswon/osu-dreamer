
from typing import Any, Callable, Union
from jaxtyping import Float

from dataclasses import dataclass

import torch as th
from torch import Tensor, nn

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

from .modules import Encoder, EncoderArgs, ChunkPad, PSVariational

@dataclass
class VQGANArgs:
    depth: int
    x_dim: int
    x_args: EncoderArgs
    a_dim: int
    a_args: EncoderArgs
    dec_args: EncoderArgs
    
class Model(pl.LightningModule):
    def __init__(
        self,

        # training parameters
        opt_args: dict[str, Any],
        step_ref: float,
        cursor_factor: float,
        kl_factor: float,

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
        self.kl_factor = kl_factor

        # model

        self.audio_encoder = nn.Sequential(
            ChunkPad(self.chunk_size, pad_value=-1.),
            MP.Conv1d(A_DIM, args.a_args.dim, 1),
            MP.PixelNorm(),
            Encoder(args.depth, args.a_args, down=True),
            MP.Conv1d(args.a_args.dim, args.a_dim, 1),
        )
        self.chart_encoder = PSVariational(args.x_args.dim, args.x_dim, nn.Sequential(
            ChunkPad(self.chunk_size),
            MP.Conv1d(X_DIM, args.x_args.dim, 1),
            MP.PixelNorm(),
            Encoder(args.depth, args.x_args, down=True),
        ))

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    MP.Conv1d(args.a_dim + args.x_dim, args.dec_args.dim, 1),
                    Encoder(args.depth, args.dec_args, down=False),
                    MP.Conv1d(args.dec_args.dim, X_DIM, 1),
                    MP.Gain(),
                )
        
            def forward(
                self,
                zx: Float[Tensor, "B D zL"],
                za: Float[Tensor, "B A zL"],
                L: int,
            ) -> Float[Tensor, str(f"B {X_DIM} L")]:
                return self.net(MP.cat([zx, MP.pixel_norm(za)], dim=1))[:,:,:L]
            
        self.decoder = Decoder()

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        chart: Float[Tensor, str(f"B {X_DIM} L")],
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        za = self.audio_encoder(audio)
        zx, kl_loss = self.chart_encoder(chart, return_loss = True)
        pred_chart = self.decoder(zx, za, L=chart.size(-1))
        recon_loss = (chart - pred_chart).pow(2).mean()
        bound_loss = (pred_chart.abs().clamp(min=1) - 1).pow(2).mean()

        cd_map = lambda chart: th.tanh(20 * (chart[..., CursorSignals, 1:] - chart[..., CursorSignals, :-1]))
        cursor_loss = (cd_map(chart) - cd_map(pred_chart)).pow(2).mean()

        
        loss = ( 
            + recon_loss
            + bound_loss
            + self.cursor_factor * cursor_loss
            + self.kl_factor * kl_loss
        )
        return loss, {
            'loss': loss.detach(),
            'recon': recon_loss.detach(),
            'cursor': cursor_loss.detach(),
            'bound': bound_loss.detach(),
            'kl': kl_loss.detach(),
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
            x_hat, _, zx_hat = self.generate(a[0], lambda _: self.chart_encoder(x), return_latents=True)
            zx_hat_rep = repeat(zx_hat, 'b d l -> b d (l r)', r=self.chunk_size)[:,:,:x.size(-1)]
            plots = [ x[0].cpu().numpy() for x in [ x, x_hat, zx_hat_rep ] ]

        exp: SummaryWriter = self.logger.experiment # type: ignore
        with plot_signals(a[0].cpu().numpy(), plots) as fig:
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
        za = repeat(za, 'a l -> b a l', b=pred_zx.size(0))
        pred_chart = self.decoder(pred_zx, za, L=audio.size(-1)).clamp(min=-1, max=1)
        if return_latents:
            return pred_chart, za, pred_zx
        return pred_chart