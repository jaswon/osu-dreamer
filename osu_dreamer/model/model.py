
from functools import partial

from jaxtyping import Float, Int

import numpy as np
import librosa

import torch as th
from torch import Tensor
import torch.nn.functional as F

from einops import repeat

try:
    import matplotlib.pyplot as plt
    USE_MATPLOTLIB = True
except:
    USE_MATPLOTLIB = False

import pytorch_lightning as pl

from osu_dreamer.data.dataset import Batch
from osu_dreamer.data.load_audio import A_DIM
from osu_dreamer.data.beatmap.encode import CursorSignals, CURSOR_DIM, X_DIM

from .diffusion import Diffusion

from .modules.encoder import Encoder, EncoderArgs
from .modules.denoiser import Denoiser, DenoiserArgs
from .modules.critic import Critic, CriticArgs
    
    
class Model(pl.LightningModule):
    def __init__(
        self,

        # validation parameters
        val_steps: int,

        # training parameters
        accum_batches: int,                 # batches to accumulate
        gen_adv_factor: float,              # adversarial loss scale factor
        optimizer: str,                     # optimizer
        denoiser_opt_args: dict[str, dict], # denoiser optimizer args
        critic_opt_args: dict[str, dict],   # critic optimizer args
        P_mean: float,
        P_std: float,

        # model hparams
        encoder_args: EncoderArgs,
        denoiser_args: DenoiserArgs,
        critic_args: CriticArgs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # model
        self.diffusion = Diffusion(P_mean, P_std)
        self.a_enc = Encoder(A_DIM, encoder_args)
        self.denoiser = Denoiser(X_DIM, encoder_args.h_dim, denoiser_args)
        self.critic = Critic(CURSOR_DIM, critic_args)

        # validation params
        self.val_steps = val_steps

        # training params
        self.accum_batches = accum_batches
        self.gen_adv_factor = gen_adv_factor
        self.optimizer = getattr(th.optim, optimizer)
        assert 'default' in denoiser_opt_args, "`default` key for `denoiser_opt_args` required"
        self.denoiser_opt_args = denoiser_opt_args
        assert 'default' in critic_opt_args, "`default` key for `critic_opt_args` required"
        self.critic_opt_args = critic_opt_args
    

    def forward(
        self, 
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        position: Int[Tensor, "B L"],
        x_real: Float[Tensor, str(f"B {X_DIM} L")]
    ) -> tuple[
        dict[str, Tensor], # log dict
        Float[Tensor, ""], # denoiser loss
        Float[Tensor, ""], # critic loss
    ]:
        # augment cursor by random flips
        x_real[:,CursorSignals] *= th.where(th.rand_like(x_real[:,CursorSignals,:1]) < .5, 1, -1)

        model = partial(self.denoiser, self.a_enc(audio), position)
        loss_weight, x_hat_uncond, x_hat_cond = self.diffusion.sample_denoised(model, x_real)
        x_fake = th.stack([ x_hat_uncond, x_hat_cond ], dim=0) # 2 B X L

        # 1. diffusion (score matching objective) loss
        diffusion_loss = (loss_weight * ( x_fake - x_real ) ** 2).mean()

        # 2. adversarial (RaSGAN objective) loss
        real_logits = self.critic(x_real[:,CursorSignals])
        fake_logits = self.critic(x_hat_cond[:,CursorSignals])
        
        real_score = real_logits - fake_logits.mean()
        fake_score = fake_logits - real_logits.mean()

        adv_denoiser_loss = F.softplus(th.stack([-fake_score, real_score])).mean()
        adv_critic_loss   = F.softplus(th.stack([-real_score, fake_score])).mean()

        denoiser_loss = diffusion_loss + adv_denoiser_loss * self.gen_adv_factor
        critic_loss = adv_critic_loss
        
        return {
            "denoiser/diffusion": diffusion_loss.detach(),
            "denoiser/adversarial": adv_denoiser_loss.detach(),
            "denoiser": denoiser_loss.detach(),
            "critic": critic_loss.detach(),
        }, denoiser_loss, critic_loss
    
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

        denoiser = partial(self.denoiser, self.a_enc(audio), p)
        return self.diffusion.sample(denoiser, None, num_steps, z, **kwargs)


#
#
# =============================================================================
# MODEL TRAINING
# =============================================================================
#
#

    def configure_optimizers(self):

        def get_param_groups(all_params, opt_args):
            params = { opt_key: [] for opt_key in opt_args }
            for p in all_params:
                opt_key = getattr(p, 'opt_key', 'default')
                params.get(opt_key, params['default']).append(p)
            return [
                { 'params': params[opt_key], **args }
                for opt_key, args in opt_args.items()
            ]

        opt_critic = self.optimizer(get_param_groups(
            self.critic.parameters(), 
            self.critic_opt_args,
        ))

        opt_denoiser = self.optimizer(get_param_groups(
            [
                *self.a_enc.parameters(),
                *self.denoiser.parameters(),
            ], 
            self.denoiser_opt_args,
        ))

        return opt_critic, opt_denoiser 

    def training_step(self, batch: Batch, batch_idx):
        opt_critic, opt_denoiser = self.optimizers() # type: ignore
        
        self.denoiser.requires_grad_(True)
        self.critic.requires_grad_(True)
        log_dict, denoiser_loss, critic_loss = self(*batch)
        self.log_dict({ f"train/{k}": v for k,v in log_dict.items() })

        self.critic.requires_grad_(False)
        self.manual_backward(denoiser_loss / self.accum_batches, retain_graph=True)

        self.critic.requires_grad_(True)
        self.denoiser.requires_grad_(False)
        self.manual_backward(critic_loss / self.accum_batches)

        if (batch_idx + 1) % self.accum_batches == 0:
            opt_critic.step()
            opt_denoiser.step()
            opt_critic.zero_grad()
            opt_denoiser.zero_grad()

    def validation_step(self, batch: Batch, batch_idx, *args, **kwargs):
        with th.no_grad():
            log_dict, *_ = self(*batch)
        self.log_dict({ f"val/{k}": v for k,v in log_dict.items() })

        if batch_idx == 0 and USE_MATPLOTLIB:
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