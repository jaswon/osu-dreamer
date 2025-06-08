
from typing import Any
from jaxtyping import Float

import torch as th
from torch import Tensor, nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.utils.tensorboard.writer import SummaryWriter

from osu_dreamer.data.beatmap.encode import X_DIM
from osu_dreamer.data.load_audio import A_DIM
from osu_dreamer.data.plot import plot_signals

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.muon import Muon
from osu_dreamer.modules.lr_schedule import make_lr_schedule, LRScheduleArgs

from .data.module import Batch

from .modules.hard_attn import HardAttn, HardAttnArgs
from .modules.ae import Encoder, Decoder

    
class Model(pl.LightningModule):
    def __init__(
        self,

        # training parameters
        opt_args: dict[str, Any],
        lr_schedule_args: LRScheduleArgs,

        # model hparams
        stride: int,                    # convolution stride
        depth: int,                     # number of strided convs
        expand: int,                    # hidden expansion factor

        emb_dim: int,                   # embedding dimension
        hard_attn_args: HardAttnArgs,

    ):
        super().__init__()
        self.save_hyperparameters()
        self.chunk_size = stride ** depth

        # training params
        self.opt_args = opt_args
        self.lr_schedule = make_lr_schedule(lr_schedule_args)

        # model
        self.hard_attn = HardAttn(emb_dim, hard_attn_args)
        self.encoder = nn.Sequential(
            MP.Conv1d(X_DIM, emb_dim, 1),
            Encoder(emb_dim, depth, stride, expand),
        )
        self.decoder = nn.Sequential(
            Decoder(emb_dim, depth, stride, expand),
            MP.Conv1d(emb_dim, X_DIM, 1),
            MP.Gain(),
        )

    def padding(self, L: int) -> int:
        """returns the amount of padding required to align a sequence of length L"""
        a = self.chunk_size
        return (a-L%a)%a

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        chart: Float[Tensor, str(f"B {X_DIM} L")],
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        
        pad = self.padding(chart.size(-1))
        if pad > 0:
            chart = F.pad(chart, (0,pad))

        z = self.encoder(chart) # B H l
        z_q, _ = self.hard_attn(z)
        pred_chart = self.decoder(z_q) # B H L

        loss = (pred_chart - chart).pow(2).mean()

        # TODO: add adversarial loss
        return loss, {
            "loss": loss.detach(),
        }
    
    @th.no_grad
    def encode(
        self,
        chart: Float[Tensor, str(f"B {X_DIM} L")],
    ) -> Float[Tensor, "B D l"]:
        pad = self.padding(chart.size(-1))
        if pad > 0:
            chart = F.pad(chart, (0, pad))
        z = self.encoder(chart) # B H l
        z_q, _ = self.hard_attn(z)
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
        opt = Muon(self.parameters(), **self.opt_args)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": th.optim.lr_scheduler.LambdaLR(opt, self.lr_schedule),
                "interval": "step",
            }
        }

    def training_step(self, batch: Batch, batch_idx):
        loss, log_dict = self(*batch)
        self.log_dict({ f"train/{k}": v for k,v in log_dict.items() })
        return loss
 
    def validation_step(self, batch: Batch, batch_idx, *args, **kwargs):
        _, log_dict = self(*batch)
        self.log_dict({ f"val/{k}": v for k,v in log_dict.items() })

        if batch_idx == 0:
            self.plot_val(batch)

    @th.no_grad()
    def plot_val(self, b: Batch):
        a,x = b
        pred_x = self.decode(self.encode(x))[:,:,:x.size(-1)]

        exp: SummaryWriter = self.logger.experiment # type: ignore
        with plot_signals(
            a[0].cpu().numpy(),
            [ x[0].cpu().numpy() for x in [ x, pred_x ] ],
        ) as fig:
            exp.add_figure("samples", fig, global_step=self.global_step)
        