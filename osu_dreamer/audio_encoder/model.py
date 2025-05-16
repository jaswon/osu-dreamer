
from typing import Any
from jaxtyping import Float, Int, Bool

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


def focal_loss(
    inputs: Float[Tensor, "B D ..."],
    target: Int[Tensor, "B ..."],
    gamma: float,
    weight: None | Float[Tensor, "D"] = None,
) -> Float[Tensor, "B ..."]:
    logpt = F.log_softmax(inputs, dim=1)
    inputs = (1 - logpt.exp()).pow(gamma) * logpt
    return F.nll_loss(inputs, target, weight, reduction='none')

    
class Model(pl.LightningModule):
    def __init__(
        self,

        # training parameters
        opt_args: dict[str, Any],
        focal_gamma: float,
        class_weights: list[float],
        mask_rate: float,

        # model hparams
        h_dim: int,
        a_dim: int,
        head_dim: int,
        encoder_args: DiTArgs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # training params
        self.opt_args = opt_args
        self.focal_gamma = focal_gamma
        self.class_weights: Tensor
        self.register_buffer('class_weights', th.tensor(class_weights).float(), persistent=False)

        assert 0 <= mask_rate <= 1
        self.mask_rate = mask_rate

        # model
        self.proj_audio = nn.Sequential(
            MP.Conv1d(A_DIM, h_dim, 1),
            DiT(h_dim, None, encoder_args),
            MP.Conv1d(h_dim, a_dim, 1),
        )

        self.mask_block = nn.Sequential(
            MP.Conv1d(a_dim+(NUM_EVENTS+1), head_dim, 1),
            DiT(head_dim, None, DiTArgs(1,1)),
        )
        self.mask_head = nn.Sequential(
            MP.Linear(head_dim, NUM_EVENTS),
            MP.Gain(),
        )

    def mask_events(self, events: Int[Tensor, "B L"]) -> tuple[Bool[Tensor, "B L"], Float[Tensor, "B L E"]]:
        a = 32 # minimum consecutive run of mask tokens

        b, L = events.size()
        l = L // a

        num_mask = (1+self.mask_rate*l) # B ~ Z[1, l] - how many tokens to mask per batch
        mask = (th.arange(l).repeat(b,1) < num_mask)[th.arange(b)[:,None], th.rand(b,l).argsort(dim=-1)] # B l

        mask = mask.repeat_interleave(a, dim=1) # B l*a
        pad = L - mask.size(1)
        if pad > 0:
            offset = int(th.randint(0, pad+1, ()))
            mask = F.pad(mask, (offset, pad-offset))

        mask_tokens = th.randint_like(events, NUM_EVENTS)       # mask with random token
        mask_tokens[th.rand(events.shape) < .8] = NUM_EVENTS    # mask with <MASK>
        mask_unchanged = th.rand(events.shape) < .1             # mask with original

        masked_events = th.where(
            (mask & ~mask_unchanged).to(events.device), 
            mask_tokens, 
            events,
        ) # B L
        masked_event_embs = F.one_hot(masked_events, NUM_EVENTS+1).float() * (NUM_EVENTS+1)**.5 # B L E+1

        return mask, masked_event_embs
    
    def pred_unmask(
        self, 
        audio_features: Float[Tensor, "B H L"], 
        masked_events: Float[Tensor, "B L E"],
    ) -> Float[Tensor, str(f"B L {NUM_EVENTS}")]:
        h = self.mask_block(MP.cat([
            audio_features,
            masked_events.transpose(1,2),
        ], dim=1))
        return self.mask_head(h.transpose(1,2))

    def forward(
        self,
        audio: Float[Tensor, str(f"B {A_DIM} L")],
        events: Int[Tensor, "B L"],
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        
        mask, masked_event_embs = self.mask_events(events)
        pred_logits = self.pred_unmask(self.proj_audio(audio), masked_event_embs)

        loss = focal_loss(
            pred_logits[mask], 
            events[mask], 
            gamma = self.focal_gamma,
            weight = self.class_weights,
        ).mean()

        return loss, {
            "loss": loss.detach(),
        }
    
    @th.no_grad
    def encode( self, audio: Float[Tensor, str(f"B {A_DIM} L")] ) -> Float[Tensor, "B H L"]:
        return self.proj_audio(audio)

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
    def plot_val(self, batch: Batch):
        a, e = batch
        _, masked_event_embs = self.mask_events(e)
        pred_logits = self.pred_unmask(self.encode(a), masked_event_embs)
        pred_e = pred_logits.argmax(dim=-1) # B L

        guides = th.arange(1, NUM_EVENTS).repeat(e.size(-1),1).T # E L

        exp: SummaryWriter = self.logger.experiment # type: ignore
        with plot_signals(
            a[0].cpu().numpy(),
            [ th.cat([guides, x[0,None].cpu()], dim=0).float().numpy() for x in [ e, pred_e ] ],
        ) as fig:
            exp.add_figure("samples", fig, global_step=self.global_step)
        