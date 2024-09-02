
from functools import partial

from typing import Any
from jaxtyping import Float

import torch as th
from torch import Tensor

from einops import repeat, rearrange

import pytorch_lightning as pl

from osu_dreamer.data.dataset import Batch
from osu_dreamer.data.load_audio import A_DIM
from osu_dreamer.data.beatmap.encode import X_DIM
from osu_dreamer.data.prepare_map import NUM_LABELS
from osu_dreamer.data.plot import plot_signals

from .adabelief import AdaBelief
from .diffusion import Diffusion

from .modules.encoder import Encoder, EncoderArgs
from .modules.denoiser import Denoiser, DenoiserArgs
from .modules.label_gen import LabelGen, LabelGenArgs

    
class Model(pl.LightningModule):
    def __init__(
        self,

        # validation parameters
        val_batches: int,
        val_steps: int,

        # training parameters
        opt_args: dict[str, Any],
        P_mean: float,
        P_std: float,

        # model hparams
        label_dim: int,
        label_gen_args: LabelGenArgs,

        audio_features: int,
        audio_encoder_args: EncoderArgs,

        denoiser_args: DenoiserArgs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # model
        self.diffusion = Diffusion(P_mean, P_std)
        self.audio_encoder = Encoder(audio_features, audio_encoder_args, in_dim=A_DIM)
        self.denoiser = Denoiser(X_DIM, audio_features, label_dim, denoiser_args)
        self.label_gen = LabelGen(label_dim, label_gen_args)

        # validation params
        self.val_batches = val_batches
        self.val_steps = val_steps

        # training params
        self.opt_args = opt_args
    

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        chart: Float[Tensor, str(f"B {X_DIM} L")],
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]: 
        c, loss_label = self.label_gen(labels)
        model = partial(self.denoiser, self.audio_encoder(audio), c)
        loss_diffusion = self.diffusion.loss(model, chart)

        loss = loss_label + loss_diffusion
        return loss, {
            "diffusion": loss_diffusion.detach(),
            "label": loss_label.detach(),
        }
    
    @th.no_grad()
    def sample(
        self, 
        audio: Float[Tensor, str(f"{A_DIM} L")],
        label: Float[Tensor, str(f"B {NUM_LABELS}")],
        num_steps: int = 0,
        **kwargs,
    ) -> tuple[
        Float[Tensor, str(f"B {X_DIM} L")],
        Float[Tensor, str(f"B {NUM_LABELS}")],
    ]:
        num_steps = num_steps if num_steps > 0 else self.val_steps

        num_samples = label.size(0)
        label_embs = self.label_gen.embedding(label).to(audio)
        pred_labels = self.label_gen.decoder(label_embs)
        
        audio = repeat(audio, 'a l -> b a l', b=num_samples)

        z = th.randn(num_samples, X_DIM, audio.size(-1), device=audio.device)

        denoiser = partial(self.denoiser, self.audio_encoder(audio), label_embs)
        pred_signal = self.diffusion.sample(denoiser, None, num_steps, z, **kwargs)

        return pred_signal, pred_labels


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
            a,x,l = batch
            bL = self.val_batches * (a.size(-1) // self.val_batches)
            a = rearrange(a[...,:bL], '1 ... (b l) -> b ... l', b = self.val_batches)
            x = rearrange(x[...,:bL], '1 ... (b l) -> b ... l', b = self.val_batches)
            l = repeat(l, '1 d -> b d', b = self.val_batches)
            _, log_dict = self(a,x,l)
        self.log_dict({ f"val/{k}": v for k,v in log_dict.items() })

        if batch_idx == 0:
            self.plot_sample(batch)

    def plot_sample(self, b: Batch):
        a_tensor, x_tensor, label_tensor = b

        with th.no_grad():
            plots = [
                x[0].cpu().numpy()
                for x in [
                    x_tensor, 
                    self.sample(a_tensor[0], label=label_tensor)[0],
                ]
            ]

        with plot_signals(a_tensor[0].cpu().numpy(), plots) as fig:
            self.logger.experiment.add_figure("samples", fig, global_step=self.global_step) # type: ignore