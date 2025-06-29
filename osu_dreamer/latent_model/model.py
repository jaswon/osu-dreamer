
from typing import Any
from jaxtyping import Float, Int

import torch as th
from torch import Tensor, nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.utilities.types import LRSchedulerPLType
from torch.utils.tensorboard.writer import SummaryWriter

from osu_dreamer.data.beatmap.encode import X_DIM
from osu_dreamer.data.load_audio import A_DIM
from osu_dreamer.data.plot import plot_signals

from osu_dreamer.modules.lr_schedule import LRScheduleArgs, make_lr_schedule
from osu_dreamer.modules.muon import Muon
import osu_dreamer.modules.mp as MP

from .data.module import Batch

from .modules.gaussian import GaussianVariationalBottleneck
from .modules.ae import Encoder, Decoder, AutoEncoderArgs
from .modules.critic import MultiScaleCritic, MultiScaleCriticArgs
from .modules.audio import AudioEncoder

@th.no_grad
def compute_perplexity(indices: Int[Tensor, "..."]) -> float:
    _, counts = th.unique(indices, return_counts=True)
    probs = counts / indices.numel()
    return th.exp(-th.sum(probs * th.log(probs.clamp(min=1e-6)))).item()

def product(factors: list[int]) -> int:
    p = 1
    for factor in factors:
        p *= factor
    return p

class Model(pl.LightningModule):
    def __init__(
        self,

        # training parameters
        opt_args: dict[str, Any],
        lr_schedule_args: LRScheduleArgs,
        grad_accum_steps: int,
        critic_steps_per_gen: int,
        gp_factor: float,
        gan_factor: float,
        fm_factor: float,
        pixel_factor: float,
        kl_factor: float,

        # model hparams
        emb_dim: int,
        h_dim: int,
        audio_h_dim: int,
        ae_args: AutoEncoderArgs,
        critic_args: MultiScaleCriticArgs,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()
        self.chunk_size = product(ae_args.strides)
        self.emb_dim = emb_dim

        # training params
        self.opt_args = opt_args
        self.lr_schedule = make_lr_schedule(lr_schedule_args)
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.critic_steps_per_gen = max(1, critic_steps_per_gen)
        self.gp_factor = gp_factor
        self.gan_factor = gan_factor
        self.fm_factor = fm_factor
        self.pixel_factor = pixel_factor
        self.kl_factor = kl_factor

        # model
        self.encoder = nn.Sequential(
            MP.Conv1d(X_DIM, h_dim, 1),
            Encoder(h_dim, ae_args),
            GaussianVariationalBottleneck(h_dim, emb_dim),
        )
        self.decoder = nn.Sequential(
            MP.PixelNorm(),
            MP.Conv1d(emb_dim, h_dim, 1),
            Decoder(h_dim, ae_args),
        )
        self.align_audio = AudioEncoder(X_DIM, h_dim, audio_h_dim)
        self.critic = MultiScaleCritic(X_DIM, critic_args)

    def padding(self, L: int) -> int:
        """returns the amount of padding required to align a sequence of length L"""
        a = self.chunk_size
        return (a-L%a)%a

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        chart: Float[Tensor, str(f"B {X_DIM} L")],
    ) -> tuple[
        Float[Tensor, str(f"B {X_DIM} L")],
        Float[Tensor, ""],
    ]:
        pad = self.padding(chart.size(-1))
        if pad > 0:
            chart = F.pad(chart, (0,pad))
        z, kl_loss = self.encoder(chart)
        return self.align_audio(audio, self.decoder(z)), kl_loss
    
    @th.no_grad
    def encode(
        self,
        chart: Float[Tensor, str(f"B {X_DIM} L")],
    ) -> Float[Tensor, "B D l"]:
        pad = self.padding(chart.size(-1))
        if pad > 0:
            chart = F.pad(chart, (0, pad))
        z, _ = self.encoder(chart)
        return z
    
    @th.no_grad
    def decode(
        self,
        a: Float[Tensor, "B A L"],
        z: Float[Tensor, "B D l"],
    ) -> Float[Tensor, str(f"B {X_DIM} L")]:
        return self.align_audio(a, self.decoder(z)).clamp(min=-1, max=1)

#
#
# =============================================================================
# MODEL TRAINING
# =============================================================================
#
#

    def configure_optimizers(self):
        c_opt = Muon(self.critic.parameters(), **self.opt_args)
        g_opt = Muon([
            *self.encoder.parameters(),
            *self.decoder.parameters(),
            *self.align_audio.parameters(),
        ], **self.opt_args)
        return [c_opt, g_opt], [
            {
                "scheduler": th.optim.lr_scheduler.LambdaLR(opt, self.lr_schedule),
                "interval": "step",
            }
            for opt in [c_opt, g_opt]
        ]

    def training_step(self, batch: Batch, batch_idx):
        opts: list[LightningOptimizer] = self.optimizers() # type: ignore
        schs: list[LRSchedulerPLType] = self.lr_schedulers() # type: ignore
        c_opt, g_opt = opts
        c_sch, g_sch = schs
        audio, chart = batch

        for i in range(self.critic_steps_per_gen):
            self._train_critic_step(audio, chart, c_opt, c_sch, batch_idx, i)
        self._train_gen_step(audio, chart, g_opt, g_sch, batch_idx)

    def _train_critic_step(self, audio: Float[Tensor, str(f"B {A_DIM} L")], true_chart: Float[Tensor, str(f"B {X_DIM} L")], c_opt, c_sch, batch_idx: int, critic_step: int):
        """Train the critic for one step"""
        B = true_chart.size(0)
        
        # train critics
        with th.no_grad():
            pred_chart = self(audio, true_chart)[0]

        critic_adv_loss = th.tensor(0., device=self.device)
        pred_all_fmaps = self.critic(pred_chart)
        true_all_fmaps = self.critic(true_chart)
        for pred_fmaps, true_fmaps in zip(pred_all_fmaps, true_all_fmaps):
            *_, pred_score = pred_fmaps
            *_, true_score = true_fmaps
            critic_adv_loss.add_( F.softplus(-true_score).mean() + F.softplus(pred_score).mean() )

        # gradient penalty
        gradient_penalty = th.tensor(0., device=self.device)
        alpha = ((th.arange(B) + th.rand(1)) / B)[:,None,None].to(self.device)
        lerp_chart = ( alpha * true_chart + (1-alpha) * pred_chart ).requires_grad_(True)
        for lerp_fmaps in self.critic(lerp_chart):
            lerp_score = lerp_fmaps[-1]
            gradients = th.autograd.grad(
                outputs=lerp_score,
                inputs=lerp_chart,
                grad_outputs=th.ones_like(lerp_score),
                create_graph=True,
                retain_graph=True,
            )[0].view(B, -1)
            gradient_penalty.add_( ((gradients.norm(2, dim=1) - 1) ** 2).mean() )

        self.manual_backward((
            + critic_adv_loss
            + self.gp_factor * gradient_penalty
        ) / self.grad_accum_steps)

        if (batch_idx*self.critic_steps_per_gen + critic_step + 1) % self.grad_accum_steps == 0:
            c_opt.step()
            c_opt.zero_grad()
            c_sch.step() # type: ignore

        if critic_step == 0:
            self.log_dict({
                "train/critic/adversarial": critic_adv_loss.detach(),
                "train/critic/gradient_penalty": gradient_penalty.detach(),
            })

    def _train_gen_step(self, audio: Float[Tensor, str(f"B {A_DIM} L")], true_chart: Float[Tensor, str(f"B {X_DIM} L")], g_opt, g_sch, batch_idx: int):
        """Train the generator for one step"""
        pred_chart, kl_loss = self(audio, true_chart)
        pixel_loss = ( pred_chart - true_chart ).pow(2).mean()

        gen_adv_loss = th.tensor(0., device=self.device)
        fm_loss = th.tensor(0., device=self.device)
        pred_all_fmaps = self.critic(pred_chart)
        with th.no_grad():
            true_all_fmaps = self.critic(true_chart)
        for pred_fmaps, true_fmaps in zip(pred_all_fmaps, true_all_fmaps):
            *pred_fmaps, pred_score = pred_fmaps
            *true_fmaps, _          = true_fmaps

            gen_adv_loss.add_( F.softplus(-pred_score).mean() )

            # feature matching
            for pred_fmap, true_fmap in zip(pred_fmaps, true_fmaps):
                fm_loss.add_( (pred_fmap - true_fmap).abs().mean() )

        self.manual_backward((
            + self.fm_factor * fm_loss 
            + self.pixel_factor * pixel_loss
            + self.gan_factor * gen_adv_loss 
            + self.kl_factor * kl_loss
        ) / self.grad_accum_steps)

        if (batch_idx + 1) % self.grad_accum_steps == 0:
            g_opt.step()
            g_opt.zero_grad()
            g_sch.step() # type: ignore

        self.log_dict({
            "train/gen/kl": kl_loss.detach(),
            "train/gen/adversarial": gen_adv_loss.detach(),
            "train/gen/reconstruction": fm_loss.detach(),
            "train/gen/l2": pixel_loss.detach(),
        })

    @th.no_grad
    def validation_step(self, batch: Batch, batch_idx, *args, **kwargs):
        if batch_idx == 0:
            self.plot_val(batch)

        audio, true_chart = batch
        pred_chart, _ = self(audio, true_chart)
        self.log_dict({
            'val/l2_rec': (true_chart - pred_chart).pow(2).mean(),
        })

    @th.no_grad
    def plot_val(self, b: Batch):
        from einops import repeat

        a,x = b
        z = self.encode(x)
        plot_z = repeat(z, 'b d l -> b d (l r)', r=self.chunk_size)[:,:,:x.size(-1)]
        pred_x = self.decode(a, z)

        exp: SummaryWriter = self.logger.experiment # type: ignore
        with plot_signals(
            a[0].cpu().numpy(),
            [ s[0].cpu().numpy() for s in [ x, pred_x, x-pred_x, plot_z ] ],
        ) as fig:
            exp.add_figure("samples", fig, global_step=self.global_step)
        