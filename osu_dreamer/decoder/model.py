
from typing import Any
from jaxtyping import Float, Int

import torch as th
from torch import Tensor, nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.utils.tensorboard.writer import SummaryWriter

from osu_dreamer.data.load_audio import A_DIM, get_frame_times
from osu_dreamer.data.plot import plot_signals

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.muon import Muon
from osu_dreamer.modules.dit import DiT, DiTArgs

from osu_dreamer.audio_encoder.model import Model as AudioEncoder

from .data.module import Batch
from .data.events import PAD, BOS, EOS, vocab_size


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
        batch_size: int,
        seq_len: int,
        opt_args: dict[str, Any],
        focal_gamma: float,

        # model hparams
        audio_encoder_ckpt: str,
        embed_dim: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        # training params
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.opt_args = opt_args
        self.focal_gamma = focal_gamma

        # model
        audio_encoder = AudioEncoder.load_from_checkpoint(audio_encoder_ckpt)
        self.audio_encoder = audio_encoder.proj_audio
        self.a_dim = audio_encoder.a_dim

        self.embed = MP.Embedding(vocab_size(), embed_dim)
        token_head = MP.Linear(embed_dim, vocab_size())
        token_head.weight = self.embed.weight
        self.token_head = nn.Sequential(
            token_head,
            MP.Gain(),
        )

        self.timing_head = nn.Sequential(
            MP.Linear(embed_dim, seq_len),
            MP.Gain(),
        )

    def make_batch(
        self,
        audio: Float[Tensor, str(f"1 {A_DIM} L")],
        tokens: Int[Tensor, "1 N"],
        timestamps: Float[Tensor, "1 N"],
    ) -> tuple[
        Float[Tensor, "B bL"],      # audio timestamps 
        Float[Tensor, "B H bL"],    # audio features
        Float[Tensor, "B bN"],      # token timestamps
        Int[Tensor, "B bN"],        # tokens
    ]:
        
        L = audio.size(-1)
        audio_features = self.audio_encoder(audio)
        frame_times = th.tensor(get_frame_times(L)).float()

        b_features = th.empty(self.batch_size, audio_features.size(1), self.seq_len) # B H bL
        b_frame_times = th.empty(self.batch_size, self.seq_len) # B bL
        b_ranges: list[tuple[int,int]] = []
        max_tokens: int = 0

        for start_idx in th.randperm(L - self.seq_len)[:self.batch_size]:
            end_idx = start_idx+self.seq_len
            b_features[start_idx] = audio_features[0,:,start_idx:end_idx]
            b_frame_times[start_idx] = frame_times[start_idx:end_idx]

            left_idx = int(th.searchsorted(timestamps, frame_times[start_idx], right=False))
            right_idx = int(th.searchsorted(timestamps, frame_times[end_idx], right=True))
            b_ranges.append((left_idx, right_idx))
            max_tokens = max(max_tokens, right_idx - left_idx)

        b_tokens = th.full((self.batch_size, max_tokens+2), PAD)
        b_timestamps = th.full((self.batch_size, max_tokens+2), -1).float()
        b_tokens[:,0] = BOS

        for idx, (left_idx, right_idx) in enumerate(b_ranges):
            num_tokens = right_idx-left_idx
            b_tokens[idx, 1:][:,:num_tokens] = tokens[0,left_idx:right_idx]
            b_tokens[idx, 1:][:,num_tokens] = EOS
            b_timestamps[idx,1:][:,:num_tokens] = timestamps[0,left_idx:right_idx]

        return b_frame_times, b_features, b_timestamps, b_tokens


    def forward(
        self,
        audio: Float[Tensor, str(f"1 {A_DIM} L")],
        tokens: Int[Tensor, "1 N"],
        timestamps: Float[Tensor, "1 N"],
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        
        b_frame_times, b_features, b_timestamps, b_tokens = self.make_batch(audio, tokens, timestamps)

        embeds = self.embed(b_tokens) # B N E

        h = self.net(
            embeds = embeds,
            embed_positions = b_timestamps,
            ctx = b_features,
            ctx_positions = b_frame_times,
        ) # B N E

        pred_logits = self.token_head(h) # B N V
        token_loss = focal_loss(
            pred_logits.transpose(1,2),
            b_tokens,
            gamma = self.focal_gamma,
        ).mean()

        # continuous ranked probability score
        pred_timings = self.timing_head(h) # B N L
        pred_timing_cdf = F.softmax(pred_timings, dim=-1).cumsum(dim=-1)
        true_timing_cdf = (b_frame_times[:,None,:] >= b_timestamps[:,:,None]).long() # B N L
        timing_loss = (pred_timing_cdf - true_timing_cdf).pow(2).mean()

        loss = token_loss + timing_loss
        return loss, {
            "loss": loss.detach(),
            "token": token_loss.detach(),
            "timing": timing_loss.detach(),
        }

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
        _, masked_events = self.mask_events(e)
        pred_logits = self.pred_unmask(self.encode(a), masked_events)
        pred_e = pred_logits.argmax(dim=-1) # B L

        guides = th.arange(1, NUM_EVENTS).repeat(e.size(-1),1).T # E L

        exp: SummaryWriter = self.logger.experiment # type: ignore
        with plot_signals(
            a[0].cpu().numpy(),
            [ th.cat([guides, x[0,None].cpu()], dim=0).float().numpy() for x in [ e, pred_e ] ],
        ) as fig:
            exp.add_figure("samples", fig, global_step=self.global_step)
        