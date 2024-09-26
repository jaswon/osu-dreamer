
from typing import Any
from jaxtyping import Float

import torch as th
from torch import Tensor

import pytorch_lightning as pl
from torch.utils.tensorboard.writer import SummaryWriter

from osu_dreamer.data.dataset import Batch
from osu_dreamer.data.load_audio import A_DIM
from osu_dreamer.data.beatmap.encode import X_DIM
from osu_dreamer.data.prepare_map import NUM_LABELS
from osu_dreamer.data.plot import plot_signals

from osu_dreamer.modules.adabelief import AdaBelief

from .vae import VAE, VAEArgs

    
class Model(pl.LightningModule):
    def __init__(
        self,

        # training parameters
        opt_args: dict[str, Any],

        # model hparams
        vae_args: VAEArgs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # model
        self.dim = vae_args.latent_dim
        self.vae = VAE(vae_args)

        # training params
        self.opt_args = opt_args
    
    def encode(self, x):
        return self.vae.encode(x)
    
    def decode(self, z):
        return self.vae.decode(z)

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        chart: Float[Tensor, str(f"B {X_DIM} L")],
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        return self.vae(chart)
#
#
# =============================================================================
# MODEL TRAINING
# =============================================================================
#
#

    def configure_optimizers(self):
        return AdaBelief(self.parameters(), **self.opt_args)

    def training_step(self, batch: Batch, batch_idx):
        loss, log_dict = self(*batch)
        self.log_dict({ f"train/{k}": v for k,v in log_dict.items() })
        return loss
 
    def validation_step(self, batch: Batch, batch_idx, *args, **kwargs):
        with th.no_grad():
            _, log_dict = self(*batch)
        self.log_dict({ f"val/{k}": v for k,v in log_dict.items() })

        if batch_idx == 0:
            self.plot_sample(batch)

    def plot_sample(self, b: Batch):
        a, x, label = b

        with th.no_grad():
            z = self.vae.encode(x)
            x_hat = self.vae.decode(z)
            plots = [ x[0].cpu().numpy() for x in [ x, x_hat, z ] ]

        with plot_signals(a[0].cpu().numpy(), plots) as fig:
            exp: SummaryWriter = self.logger.experiment # type: ignore
            exp.add_figure("samples", fig, global_step=self.global_step)