
from functools import partial

from jaxtyping import Float, Int

import numpy as np

import torch as th
from torch import nn, Tensor

from einops import repeat, rearrange

import pytorch_lightning as pl

from osu_dreamer.data.dataset import Batch
from osu_dreamer.data.load_audio import A_DIM
from osu_dreamer.data.beatmap.encode import X_DIM
from osu_dreamer.data.plot import plot_signals

from .diffusion import Diffusion

from .modules.encoder import Encoder, EncoderArgs
from .modules.denoiser import Denoiser, DenoiserArgs
    
    
class Model(pl.LightningModule):
    def __init__(
        self,

        # validation parameters
        val_batches: int,
        val_steps: int,

        # training parameters
        optimizer: str,                     # optimizer
        opt_args: dict[str, dict],          # optimizer args
        P_mean: float,
        P_std: float,

        # model hparams
        audio_features: int,
        audio_encoder_args: EncoderArgs,
        denoiser_args: DenoiserArgs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # model
        self.diffusion = Diffusion(P_mean, P_std)

        self.audio_encoder = nn.Sequential(
            nn.Conv1d(A_DIM, audio_features, 1),
            Encoder(audio_features, audio_encoder_args)
        )
        self.denoiser = Denoiser(X_DIM, audio_features, denoiser_args)

        # validation params
        self.val_batches = val_batches
        self.val_steps = val_steps

        # training params
        self.optimizer = getattr(th.optim, optimizer)
        assert 'default' in opt_args, "`default` key for `opt_args` required"
        self.opt_args = opt_args
    

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        position: Int[Tensor, "B L"],
        x: Float[Tensor, str(f"B {X_DIM} L")],
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]: 
            
        model = partial(self.denoiser, self.audio_encoder(audio), position)
        loss = self.diffusion.loss(model, x)
        return loss, { "diffusion": loss.detach() }
    
    @th.no_grad()
    def sample(
        self, 
        audio: Float[Tensor, str(f"{A_DIM} L")],
        num_samples: int = 1,
        num_steps: int = 0,
        **kwargs,
    ) -> Float[Tensor, str(f"B {X_DIM} L")]:
        l = audio.size(-1)
        audio = repeat(audio, 'a l -> b a l', b=num_samples)
        p = repeat(th.arange(l), 'l -> b l', b=num_samples).to(audio.device)

        num_steps = num_steps if num_steps > 0 else self.val_steps

        z = th.randn(num_samples, X_DIM, l, device=audio.device)

        denoiser = partial(self.denoiser, self.audio_encoder(audio), p)
        return self.diffusion.sample(denoiser, None, num_steps, z, **kwargs)


#
#
# =============================================================================
# MODEL TRAINING
# =============================================================================
#
#

    def configure_optimizers(self):

        def get_param_groups(all_params: list[Tensor], opt_args: dict[str, dict]):
            params = { opt_key: [] for opt_key in opt_args }
            for p in all_params:
                opt_key = getattr(p, 'opt_key', 'default')
                params.get(opt_key, params['default']).append(p)
            return [
                { 
                    'params': params[opt_key], 
                    **({} if opt_key == "default" else args),
                }
                for opt_key, args in opt_args.items()
            ]
        
        return self.optimizer(get_param_groups(
            [
                *self.denoiser.parameters(), 
                *self.audio_encoder.parameters(),
            ], 
            self.opt_args,
        ), **self.opt_args['default'])

    def training_step(self, batch: Batch, batch_idx):
        loss, log_dict = self(*batch)
        self.log_dict({ f"train/{k}": v for k,v in log_dict.items() })
        return loss
 
    def validation_step(self, batch: Batch, batch_idx, *args, **kwargs):
        with th.no_grad():
            a,p,x = batch
            bL = self.val_batches * (a.size(-1) // self.val_batches)
            a = rearrange(a[...,:bL], '1 ... (b l) -> b ... l', b = self.val_batches)
            p = rearrange(p[...,:bL], '1 ... (b l) -> b ... l', b = self.val_batches)
            x = rearrange(x[...,:bL], '1 ... (b l) -> b ... l', b = self.val_batches)
            _, log_dict = self(a,p,x)
        self.log_dict({ f"val/{k}": v for k,v in log_dict.items() })

        if batch_idx == 0:
            self.plot_sample(batch)

    def plot_sample(self, b: Batch):
        a_tensor, _, x_tensor = b
        
        a: Float[np.ndarray, "A L"] = a_tensor.squeeze(0).cpu().numpy()

        with th.no_grad():
            plots = [
                x.squeeze(0).cpu().numpy()
                for x in [
                    x_tensor, 
                    self.sample(a_tensor.squeeze(0)),
                ]
            ]

        with plot_signals(a, plots) as fig:
            self.logger.experiment.add_figure("samples", fig, global_step=self.global_step) # type: ignore