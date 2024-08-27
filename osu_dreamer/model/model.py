
from jaxtyping import Float

import numpy as np

import torch as th
from torch import Tensor
import torch.nn.functional as F

import pytorch_lightning as pl

from einops import repeat

from osu_dreamer.data.dataset import Batch
from osu_dreamer.data.load_audio import A_DIM
from osu_dreamer.data.beatmap.encode import X_DIM
from osu_dreamer.data.plot import plot_signals

from .adabelief import AdaBelief
from .modules.critic import Critic, CriticArgs
from .modules.generator import Generator, GeneratorArgs


class Model(pl.LightningModule):
    def __init__(
        self,

        # model hparams
        generator_args: GeneratorArgs,
        critic_args: CriticArgs,

        # training parameters
        critic_lr: float,
        gen_lr: float,
        r1_gamma: float,
        gen_adv_factor: float,
        real_noise: float,
        critic_steps: int = 1,
        gen_steps: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # model
        self.generator = Generator(X_DIM, A_DIM, generator_args)
        self.critic = Critic(X_DIM, A_DIM, critic_args) # reports P(X in P_data)
    
        # training params
        self.critic_lr = critic_lr
        self.gen_lr = gen_lr
        self.gen_adv_factor = gen_adv_factor
        self.r1_gamma = r1_gamma
        self.real_noise = real_noise
        self.gen_steps = gen_steps
        self.critic_steps = critic_steps
    
    @th.no_grad()
    def sample(self, a: Float[Tensor, "A L"], num_samples: int = 1) -> Float[Tensor, "B X L"]:
        return self.generator(repeat(a, 'a l -> b a l', b=num_samples))

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
            betas=(.5,.999),
        )

        opt_gen = AdaBelief(
            self.generator.parameters(), 
            lr=self.gen_lr,
            betas=(.5,.999),
        )

        return [opt_crit, opt_gen]
    
    def adversarial_critic_loss(
        self, 
        logits_real: Float[Tensor, "B l"], 
        logits_fake: Float[Tensor, "B l"],
    ) -> Float[Tensor, ""]:
        """relativistic average GAN loss"""

        real_score = logits_real - logits_fake.mean()
        fake_score = logits_fake - logits_real.mean()

        # return th.stack([real_score-1, fake_score+1]).pow(2).mean() # LSGAN
        return F.softplus(th.stack([-real_score, fake_score])).mean() # Softplus adaptation

    def training_step(self, batch: Batch, batch_idx):
        opt_crit, opt_gen = self.optimizers() # type: ignore
        a, x_real = batch

        # noise `x_real` to handicap critic
        x_real = x_real + th.randn_like(x_real) * self.real_noise

        #################### Train Critic ####################

        self.critic.requires_grad_(True)
        self.generator.requires_grad_(False)
        x_real.requires_grad_(True)
        x_fake = self.generator(a).detach()
        for _ in range(self.critic_steps):
            logits_real = self.critic(a, x_real)
            logits_fake = self.critic(a, x_fake)

            critic_adv_loss = self.adversarial_critic_loss(logits_real, logits_fake)

            # R1 Regularization
            r1_grad = th.autograd.grad(
                outputs=logits_real.float(),
                inputs=x_real.float(),
                grad_outputs=th.ones_like(logits_real.float()),
                create_graph=True,
            )[0]
            r1_loss = 0.5 * r1_grad.pow(2).sum((1,2)).mean()

            critic_loss = critic_adv_loss + r1_loss * self.r1_gamma
            if critic_loss.isnan():
                raise RuntimeError('critic nan loss')
            
            self.manual_backward(critic_loss)
            opt_crit.step()
            opt_crit.zero_grad()

        self.log('train/critic/adv', critic_adv_loss.detach())
        self.log('train/critic/r1', r1_loss.detach())

        #################### Train Generator ####################

        self.critic.requires_grad_(False)
        self.generator.requires_grad_(True)
        x_real.requires_grad_(False)
        logits_real = self.critic(a, x_real).detach()
        for _ in range(self.gen_steps):
            x_fake = self.generator(a)
            logits_fake = self.critic(a, x_fake)
            gen_adv_loss = self.adversarial_critic_loss(logits_fake, logits_real)

            # reconstruction loss for low frequency structure
            gen_recon_loss = F.mse_loss(x_real, x_fake)

            gen_loss = gen_recon_loss + gen_adv_loss * self.gen_adv_factor
            if gen_loss.isnan():
                raise RuntimeError('generator nan loss')
            
            self.manual_backward(gen_loss)
            opt_gen.step()
            opt_gen.zero_grad()

        self.log('train/gen/adv', gen_adv_loss.detach())
        self.log('train/gen/recon', gen_recon_loss.detach())

    def validation_step(self, batch: Batch, batch_idx, *args, **kwargs):
        if batch_idx == 0:
            self.plot_sample(batch)

    def plot_sample(self, b: Batch):
        a_tensor, x_tensor = b
        
        a: Float[np.ndarray, "A L"] = a_tensor[0].cpu().numpy()

        with th.no_grad():
            plots = [
                x[0].cpu().numpy()
                for x in [
                    x_tensor + th.randn_like(x_tensor) * self.real_noise, 
                    self.generator(a_tensor),
                ]
            ]

        with plot_signals(a, plots) as fig:
            self.logger.experiment.add_figure("samples", fig, global_step=self.global_step) # type: ignore