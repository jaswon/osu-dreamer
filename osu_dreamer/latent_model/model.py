
from typing import Any
from jaxtyping import Float

from dataclasses import dataclass

import torch as th
from torch import Tensor, nn

import pytorch_lightning as pl
from torch.utils.tensorboard.writer import SummaryWriter

from osu_dreamer.data.dataset import Batch
from osu_dreamer.data.load_audio import A_DIM
from osu_dreamer.data.beatmap.encode import X_DIM, CursorSignals, BeatmapEncoding
from osu_dreamer.data.prepare_map import NUM_LABELS
from osu_dreamer.data.plot import plot_signals

from osu_dreamer.modules.adabelief import AdaBelief

from osu_dreamer.modules.wavenet import WaveNet, WaveNetArgs

class Residual(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return x + self.net(x)

@dataclass
class VAEArgs:
    beta: float

    latent_dim: int
    h_dim: int
    
    wavenet_args: WaveNetArgs
    
class Model(pl.LightningModule):
    def __init__(
        self,

        # training parameters
        opt_args: dict[str, Any],
        latent_noise: float, 
        slider_importance_factor: float,

        # model hparams
        args: VAEArgs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.dim = args.latent_dim

        # training params
        self.opt_args = opt_args
        self.latent_noise = latent_noise
        self.slider_importance_factor = slider_importance_factor

        # model
        self.beta = args.beta

        block = lambda dim: Residual(nn.Sequential(
            nn.Conv1d(dim, dim, 3,1,1, groups=dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),
        ))

        self.encoder = nn.Sequential(
            nn.Conv1d(X_DIM, args.h_dim, 1),
            WaveNet(args.h_dim, args.wavenet_args, block, transpose=False),
            nn.Conv1d(args.h_dim, args.latent_dim * 2, 1),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(args.latent_dim, args.h_dim, 1),
            WaveNet(args.h_dim, args.wavenet_args, block, transpose=True),
            nn.Conv1d(args.h_dim, X_DIM, 1),
        )

    def _encoder(
        self, 
        x: Float[Tensor, str(f"B {X_DIM} L")]
    ) -> tuple[
        Float[Tensor, "B Z L"], # mean
        Float[Tensor, "B Z L"], # logvar
    ]:
        return self.encoder(x).chunk(2, dim=1)

    def _reparam(
        self,
        mean: Float[Tensor, "B Z L"],
        logvar: Float[Tensor, "B Z L"],
    ) -> Float[Tensor, "B Z L"]:
        return mean + th.randn_like(mean) * th.exp(logvar * .5)
    
    def _decoder(
        self, 
        z: Float[Tensor, "B Z L"]
    ) -> Float[Tensor, str(f"B {X_DIM} L")]:
        return self.decoder(z)

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        x: Float[Tensor, str(f"B {X_DIM} L")],
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        mean, logvar = self._encoder(x)
        kl_loss = .5 * (mean ** 2 + logvar.exp() - logvar - 1).sum(dim=1).mean()

        z = self._reparam(mean, logvar)
        x_hat = self._decoder(z + th.randn_like(z) * self.latent_noise)
        recon_loss = th.mean((x - x_hat) ** 2)
        bound_loss = th.mean((x_hat.abs().clamp(min=1) - 1) ** 2)

        x_cursor_diff = x[:, CursorSignals, 1:] - x[:, CursorSignals, :-1]
        x_hat_cursor_diff = x_hat[:, CursorSignals, 1:] - x_hat[:, CursorSignals, :-1]

        # cursor diffs are more important during sliders
        sustaining = (x[:,[BeatmapEncoding.SLIDER],1:]+1)/2
        cursor_diff_factor = 1 + (self.slider_importance_factor-1) * sustaining
        cursor_diff_loss = th.mean(cursor_diff_factor * (x_cursor_diff - x_hat_cursor_diff) ** 2)

        loss = recon_loss + cursor_diff_loss + bound_loss + self.beta * kl_loss
        return loss, {
            'loss': loss.detach(),
            'recon': recon_loss.detach(),
            'cursor': cursor_diff_loss.detach(),
            'bound': bound_loss.detach(),
            'kl': kl_loss.detach(),
        }

    def encode(
        self, 
        x: Float[Tensor, str(f"B {X_DIM} L")]
    ) -> Float[Tensor, "B Z L"]:
        return self._reparam(*self._encoder(x))

    def decode(
        self, 
        z: Float[Tensor, "B Z L"]
    ) -> Float[Tensor, str(f"B {X_DIM} L")]:
        return self._decoder(z).clamp(min=-1, max=1)
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
            z = self.encode(x)
            x_hat = self.decode(z)
            plots = [ x[0].cpu().numpy() for x in [ x, x_hat, z ] ]

        with plot_signals(a[0].cpu().numpy(), plots) as fig:
            exp: SummaryWriter = self.logger.experiment # type: ignore
            exp.add_figure("samples", fig, global_step=self.global_step)