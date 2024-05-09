
from functools import partial

from jaxtyping import Float, Int

import numpy as np
import librosa

import torch as th
from torch import Tensor

from einops import repeat

try:
    import matplotlib.pyplot as plt
    USE_MATPLOTLIB = True
except:
    USE_MATPLOTLIB = False

import pytorch_lightning as pl

from osu_dreamer.data.dataset import Batch
from osu_dreamer.data.load_audio import A_DIM
from osu_dreamer.data.beatmap.encode import CURSOR_DIM, HIT_DIM

from .diffusion import Diffusion

from .modules.encoder import Encoder, EncoderArgs
from .modules.denoiser import Denoiser, DenoiserArgs
    
    
class Model(pl.LightningModule):
    def __init__(
        self,

        # validation parameters
        val_steps: int,

        # training parameters
        optimizer: str,                     # optimizer
        optimizer_args: dict[str, dict],    # optimizer args
        P_mean: float,
        P_std: float,

        # model hparams
        encoder_args: EncoderArgs,
        denoiser_args: DenoiserArgs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # model
        self.diffusion = Diffusion(P_mean, P_std)
        self.a_enc = Encoder(A_DIM, encoder_args)
        self.denoiser = Denoiser(HIT_DIM+CURSOR_DIM, encoder_args.h_dim, denoiser_args)

        # validation params
        self.val_steps = val_steps

        # training params
        self.optimizer = getattr(th.optim, optimizer)
        assert 'default' in optimizer_args, "`default` key for `optimizer_args` required"
        self.optimizer_args = optimizer_args

    def forward(
        self, 
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        position: Int[Tensor, "B L"],
        hit_sig: Float[Tensor, str(f"B {HIT_DIM} L")],
        cursor_sig: Float[Tensor, str(f"B {CURSOR_DIM} L")],
    ) -> tuple[
        dict[str, Tensor], # log dict
        Float[Tensor, ""], # loss
    ]:
        # augment cursor by random flips
        cursor_sig *= th.where(th.rand_like(cursor_sig[...,:1]) < .5, 1, -1)

        x = th.cat([hit_sig, cursor_sig], dim=1)

        model = partial(self.denoiser, self.a_enc(audio), position)
        loss_weight, x_hat_uncond, x_hat_cond = self.diffusion.sample_denoised(model, x)
        x_hat = th.stack([ x_hat_uncond, x_hat_cond ], dim=0)
        loss = (loss_weight * ( x_hat - x ) ** 2).mean()

        return {
            'loss': loss.detach(),
        }, loss
    
    @th.no_grad()
    def sample(
        self, 
        a: Float[Tensor, str(f"{A_DIM} L")],
        num_samples: int = 1,
        num_steps: int = 0,
        **kwargs,
    ) -> Float[Tensor, str(f"B {HIT_DIM + CURSOR_DIM} L")]:
        l = a.size(-1)
        a = repeat(a, 'a l -> b a l', b=num_samples)
        p = repeat(th.arange(l), 'l -> b l', b=num_samples).to(a.device)

        num_steps = num_steps if num_steps > 0 else self.val_steps

        z = th.randn(num_samples, HIT_DIM + CURSOR_DIM, l, device=a.device)
        model = partial(self.denoiser, self.a_enc(a), p)
        return self.diffusion.sample(model, num_steps, z, **kwargs)


#
#
# =============================================================================
# MODEL TRAINING
# =============================================================================
#
#

    def configure_optimizers(self):

        params = { opt_key: [] for opt_key in self.optimizer_args }
        for p in self.parameters():
            opt_key = getattr(p, 'opt_key', 'default')
            params.get(opt_key, params['default']).append(p)

        opt = self.optimizer([
            { 'params': params[opt_key], **args }
            for opt_key, args in self.optimizer_args.items()
        ])

        return opt

    def training_step(self, batch: Batch, batch_idx):
        log_dict, loss = self(*batch)
        self.log_dict({ f"train/{k}": v for k,v in log_dict.items() })
        return loss

    def validation_step(self, batch: Batch, batch_idx, *args, **kwargs):
        with th.no_grad():
            log_dict, _ = self(*batch)
        self.log_dict({ f"val/{k}": v for k,v in log_dict.items() })

        if batch_idx == 0 and USE_MATPLOTLIB:
            self.plot_sample(batch)

        th.cuda.empty_cache()

    def plot_sample(self, b: Batch):
        a_tensor, _, hit_tensor, cursor_tensor = b
        x_tensor = th.cat([hit_tensor, cursor_tensor], dim=1)
        
        a: Float[np.ndarray, "A L"] = a_tensor.squeeze(0).cpu().numpy()

        with th.no_grad():
            plots = [
                x.squeeze(0).cpu().numpy()
                for x in [
                    x_tensor, 
                    self.sample(a_tensor.squeeze(0)),
                ]
            ]
        
        margin, margin_left = .1, .5
        height_ratios = [.8] + [.6] * len(plots)
        plots_per_row = len(height_ratios)
        w, h = a.shape[-1] * .01, sum(height_ratios) * .4

        # split plot across multiple rows
        split = ((w/h)/(3/5)) ** .5 # 3 wide by 5 tall aspect ratio
        split = int(split + 1)
        w = w // split
        h = h * split
        height_ratios = height_ratios * split
        
        fig, all_axs = plt.subplots(
            len(height_ratios), 1,
            figsize=(w, h),
            sharex=True,
            gridspec_kw=dict(
                height_ratios=height_ratios,
                hspace=.1,
                left=margin_left/w,
                right=1-margin/w,
                top=1-margin/h,
                bottom=margin/h,
            )
        )

        win_len = a.shape[-1] // split
        for i in range(split):
            ax1, *axs = all_axs[i * plots_per_row: (i+1) * plots_per_row]
            sl = (..., slice(i * win_len, (i+1) * win_len))

            ax1.imshow(librosa.power_to_db(a[sl]), origin="lower", aspect='auto')
            
            for (i, sample), ax in zip(enumerate(plots), axs):
                ax.margins(x=0)
                for ch in sample[sl]:
                    ax.plot(ch)

        self.logger.experiment.add_figure("samples", fig, global_step=self.global_step) # type: ignore
        plt.close(fig)