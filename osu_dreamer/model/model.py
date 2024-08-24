
from typing import Optional
from jaxtyping import Float, Int

import numpy as np

import torch as th
from torch import Tensor
import torch.nn.functional as F

import pytorch_lightning as pl

from osu_dreamer.data.dataset import Batch
from osu_dreamer.data.load_audio import A_DIM
from osu_dreamer.data.beatmap.encode import X_DIM
from osu_dreamer.data.plot import plot_signals

from osu_dreamer.common.adabelief import AdaBelief

from .modules.critic import Critic, CriticArgs
from .modules.generator import Generator, GeneratorArgs


class Model(pl.LightningModule):
    def __init__(
        self,

        # training parameters
        accum_batches: int,
        critic_lr: float,
        gen_lr: float,
        r1_gamma: float,
        gen_adv_factor: float,
        grad_clip_threshold: float,

        # model hparams
        generator_args: GeneratorArgs,
        critic_args: CriticArgs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # model
        self.generator = Generator(X_DIM, A_DIM, generator_args)
        self.critic = Critic(X_DIM, A_DIM, critic_args) # reports P(X in P_data)
    
        # training params
        self.accum_batches = accum_batches
        self.critic_lr = critic_lr
        self.gen_lr = gen_lr
        self.gen_adv_factor = gen_adv_factor
        self.r1_gamma = r1_gamma
        self.grad_clip_threshold = grad_clip_threshold
    
    def sample(
        self, 
        a: Float[Tensor, "A L"],
        p: Int[Tensor, "L"],
        z: Optional[Float[Tensor, "Z"]] = None,
    ) -> Float[Tensor, "X L"]:
        z = z[None] if z is not None else None
        return self.generator(a[None], p[None], z=z)[0]

    def forward(
        self,
        a: Float[Tensor, "B A L"],
        p: Int[Tensor, "B L"],
        x_real: Float[Tensor, "B X L"],
    ):
        self.generator.requires_grad_(True)
        self.critic.requires_grad_(True)

        x_fake = self.generator(a, p)

        with th.enable_grad():
            x_real.requires_grad_()
            logits_real = self.critic(a, p, x_real)
        logits_fake = self.critic(a, p, x_fake)

        # RaSGAN adversarial loss
        real_score = logits_real - logits_fake.mean()
        fake_score = logits_fake - logits_real.mean()

        adv_loss_g = F.softplus(th.stack([-fake_score, real_score])).mean()
        adv_loss_c = F.softplus(th.stack([-real_score, fake_score])).mean()

        #################### 1. Compute Generator Loss ####################

        # reconstruction loss for low frequency structure
        gen_recon_loss = F.mse_loss(x_real, x_fake)

        gen_loss = gen_recon_loss + adv_loss_g * self.gen_adv_factor

        if gen_loss.isnan():
            raise RuntimeError('generator nan loss')

        #################### 2. Compute Critic Loss ####################

        # R1 Regularization
        with th.enable_grad():
            r1_grad = th.autograd.grad(
                outputs=logits_real,
                inputs=x_real,
                grad_outputs=th.ones_like(logits_real),
                create_graph=True,
            )[0]
        r1_loss = 0.5 * r1_grad.pow(2).sum((1,2)).mean()

        critic_loss = adv_loss_c + r1_loss * self.r1_gamma

        if critic_loss.isnan():
            raise RuntimeError('critic nan loss')
        
        return gen_loss, critic_loss, {
            'gen/recon': gen_recon_loss.detach(),
            'gen/adv': adv_loss_g.detach(),
            'critic/adv': adv_loss_c.detach(),
            'critic/r1': r1_loss.detach(),
        }

#
#
# =============================================================================
# MODEL TRAINING
# =============================================================================
#
#

    def configure_optimizers(self):
        opt_crit = AdaBelief(
            self.critic.parameters(),
            lr=self.critic_lr,
        )

        opt_gen = AdaBelief(
            self.generator.parameters(), 
            lr=self.gen_lr,
        )

        return [opt_crit, opt_gen]

    def training_step(self, batch: Batch, batch_idx):
        opt_crit, opt_gen = self.optimizers() # type: ignore
        should_step = (batch_idx + 1) % self.accum_batches == 0

        self.generator.requires_grad_(True)
        self.critic.requires_grad_(True)
        gen_loss, critic_loss, log_dict = self(*batch)

        self.log_dict({ f"train/{k}":v for k,v in log_dict.items() })

        self.critic.requires_grad_(False)
        self.manual_backward(gen_loss / self.accum_batches, retain_graph=True)

        self.critic.requires_grad_(True)
        self.generator.requires_grad_(False)
        self.manual_backward(critic_loss / self.accum_batches)

        if should_step:
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip_threshold)
            opt_crit.step()
            opt_crit.zero_grad()
            th.nn.utils.clip_grad_norm_(self.generator.parameters(), self.grad_clip_threshold)
            opt_gen.step()
            opt_gen.zero_grad()

    def validation_step(self, batch: Batch, batch_idx, *args, **kwargs):

        with th.no_grad():
            _, _, log_dict = self(*batch)
        self.log_dict({ f"val/{k}":v for k,v in log_dict.items() })

        if batch_idx == 0:
            self.plot_sample(batch)

    def plot_sample(self, b: Batch):
        a_tensor, p_tensor, x_tensor = b
        
        a: Float[np.ndarray, "A L"] = a_tensor[0].cpu().numpy()

        with th.no_grad():
            plots = [
                x.cpu().numpy()
                for x in [
                    x_tensor[0], 
                    self.sample(a_tensor[0], p_tensor[0]),
                ]
            ]

        with plot_signals(a, plots) as fig:
            self.logger.experiment.add_figure("samples", fig, global_step=self.global_step) # type: ignore