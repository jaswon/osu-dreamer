
from typing import Any
from jaxtyping import Float, Int

import torch as th
from torch import Tensor, nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.utils.tensorboard.writer import SummaryWriter

from osu_dreamer.data.load_audio import A_DIM
from osu_dreamer.data.plot import plot_signals

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.muon import Muon
from osu_dreamer.modules.dit import DiT, DiTArgs

from .data.dataset import Batch
from .data.events import NUM_EVENTS

from .modules.vq import VectorQuantizer, VQArgs
from .modules.focal import focal_loss

    
class Model(pl.LightningModule):
    def __init__(
        self,

        # training parameters
        opt_args: dict[str, Any],
        focal_gamma: float,
        class_weights: list[float],

        # model hparams
        stride: int,        # convolution stride
        depth: int,         # number of strided convs

        n_embs: int,        # size of vq vocab
        emb_dim: int,         # dimension of vq embs
        vq_args: VQArgs,    # vq args

        audio_args: DiTArgs,
        mix_args: DiTArgs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.chunk_size = stride ** depth

        # model
        self.proj_audio = nn.Sequential(
            MP.Conv1d(A_DIM, emb_dim, 1),
            DiT(emb_dim, None, audio_args),
        )
        self.enc_audio = MP.Conv1d(emb_dim, emb_dim, 1)
        self.dec_audio = MP.Conv1d(emb_dim, emb_dim, 1)

        self.emb = MP.Embedding(NUM_EVENTS, emb_dim)
        self.head = MP.Linear(emb_dim, NUM_EVENTS)
        self.head.weight = self.emb.weight

        self.vq = VectorQuantizer(n_embs, emb_dim, vq_args)
        self.enc_mix = DiT(emb_dim, None, mix_args)
        self.dec_mix = DiT(emb_dim, None, mix_args)
        self.down = nn.Sequential(*(
            layer for _ in range(depth)
            for layer in [
                MP.SiLU(),
                nn.Conv1d(
                    emb_dim, emb_dim, 
                    stride+2, stride, 1,
                    groups=emb_dim, bias=False,
                ),
                MP.Conv1d(emb_dim, emb_dim, 1),
            ]
        ))
        self.up = nn.Sequential(*(
            layer for _ in range(depth)
            for layer in [
                MP.SiLU(),
                nn.ConvTranspose1d(
                    emb_dim, emb_dim, 
                    stride+2, stride, 1,
                    groups=emb_dim, bias=False,
                ),
                MP.Conv1d(emb_dim, emb_dim, 1),
            ]
        ))

        # training params
        self.opt_args = opt_args
        self.focal_gamma = focal_gamma
        self.class_weights: Tensor
        self.register_buffer('class_weights', th.tensor(class_weights).float(), persistent=False)

    def padding(self, L: int) -> int:
        """returns the amount of padding required to align a sequence of length L"""
        a = self.chunk_size
        return (a-L%a)%a

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        events: Int[Tensor, "B L"],
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        
        pad = self.padding(events.size(-1))
        if pad > 0:
            audio = F.pad(audio, (0,pad))
            events = F.pad(events, (0,pad))

        afeats = self.proj_audio(audio) # B A L

        embs = self.emb(events).transpose(1,2) # B H L
        z = self.down(self.enc_mix(embs + self.enc_audio(afeats))) # B H l
        z_q, _, vq_loss = self.vq(z)
        h = self.dec_mix(self.up(z_q) + self.dec_audio(afeats)) # B H L
        pred_logits = self.head(h.transpose(1,2)).transpose(1,2)

        ce_loss = focal_loss(
            pred_logits, 
            events, 
            gamma = self.focal_gamma,
            weight = self.class_weights,
        ).mean()

        loss = vq_loss + ce_loss
        return loss, {
            "vq": vq_loss.detach(),
            "ce": ce_loss.detach(),
            "loss": loss.detach(),
        }
    
    @th.no_grad
    def encode(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        events: Int[Tensor, "B L"],
    ) -> Int[Tensor, "B l"]:
        pad = self.padding(events.size(-1))
        if pad > 0:
            audio = F.pad(audio, (0, pad))
            events = F.pad(events, (0, pad))
        embs = self.emb(events).transpose(1,2) # B H L
        afeats = self.proj_audio(audio) # B A L
        z = self.down(self.enc_mix(embs + self.enc_audio(afeats))) # B H l
        _, inds, _ = self.vq(z)
        return inds
    
    @th.no_grad
    def decode(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        inds: Int[Tensor, "B l"],
    ) -> Int[Tensor, "B L"]:
        pad = self.padding(audio.size(-1))
        if pad > 0:
            audio = F.pad(audio, (0, pad))
        afeats = self.proj_audio(audio) # B A L
        z_q = self.vq.lookup(inds) # B H L
        h = self.dec_mix(self.up(z_q) + self.dec_audio(afeats)) # B H L
        pred_logits = self.head(h.transpose(1,2)).transpose(1,2)
        if pad > 0:
            pred_logits = pred_logits[:,:,:-pad]
        return pred_logits.argmax(dim=1)

#
#
# =============================================================================
# MODEL TRAINING
# =============================================================================
#
#

    def configure_optimizers(self):
        return Muon(self.parameters(), **self.opt_args)

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
        a, e = b
        pred_e = self.decode(a, self.encode(a, e))

        guides = th.arange(1, NUM_EVENTS).repeat(e.size(-1),1).T # E L

        exp: SummaryWriter = self.logger.experiment # type: ignore
        with plot_signals(
            a[0].cpu().numpy(),
            [ th.cat([guides, x[0,None].cpu()], dim=0).float().numpy() for x in [ e, pred_e ] ],
        ) as fig:
            exp.add_figure("samples", fig, global_step=self.global_step)
        