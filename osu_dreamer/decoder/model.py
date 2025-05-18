
import time
from typing import Any
from jaxtyping import Float, Int, Shaped

import torch as th
from torch import Tensor, nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.utils.tensorboard.writer import SummaryWriter

from osu_dreamer.data.labels import NUM_LABELS
from osu_dreamer.data.load_audio import A_DIM, get_frame_times

import osu_dreamer.modules.mp as MP
from osu_dreamer.modules.muon import Muon

from osu_dreamer.audio_encoder.model import Model as AudioEncoder

from .data.module import Batch
from .data.events import PAD, BOS, EOS, decode, vocab_size, DIFF

from .modules.label import LabelEmbedding, LabelEmbeddingArgs
from .modules.decoder import Decoder, DecoderArgs


def focal_loss(
    inputs: Float[Tensor, "B D ..."],
    target: Int[Tensor, "B ..."],
    gamma: float,
    weight: None | Float[Tensor, "D"] = None,
) -> Float[Tensor, "B ..."]:
    logpt = F.log_softmax(inputs, dim=1)
    inputs = (1 - logpt.exp()).pow(gamma) * logpt
    return F.nll_loss(inputs, target, weight, reduction='none')

def roll_by_shifts(
    input: Shaped[Tensor, "B N"], 
    shifts: Int[Tensor, "B"],
) -> Shaped[Tensor, "B N"]:
    b,n = input.size()
    col_idxs = th.arange(n).repeat(b,1)
    shifted_idxs = (col_idxs - shifts[:,None]) % n
    return th.gather(input, 1, shifted_idxs.long())

    
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

        label_dim: int,
        label_emb_args: LabelEmbeddingArgs,

        decoder_args: DecoderArgs,
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

        self.label_emb = LabelEmbedding(label_dim, label_emb_args)
        self.label_head = nn.Linear(embed_dim, label_dim)

        self.decoder = Decoder(
            embed_dim,
            self.a_dim,
            label_dim,
            decoder_args,
            self.seq_len,
        )

    def make_batch(
        self,
        audio: Float[Tensor, str(f"1 {A_DIM} L")],
        tokens: Int[Tensor, "1 N"],
        timestamps: Float[Tensor, "1 N"],
    ) -> tuple[
        Float[Tensor, "B bL"],      # audio timestamps 
        Float[Tensor, "B bL H"],    # audio features
        Float[Tensor, "B bN"],      # token timestamps
        Int[Tensor, "B bN"],        # tokens
    ]:
        
        L = audio.size(-1)
        audio_features = self.audio_encoder(audio).transpose(1,2)
        frame_times = th.tensor(get_frame_times(L)).float()

        b_features = th.empty(self.batch_size, self.seq_len, audio_features.size(-1)) # B H bL
        b_frame_times = th.empty(self.batch_size, self.seq_len) # B bL
        b_ranges: list[tuple[int,int]] = []
        max_tokens: int = 0

        for start_idx in th.randperm(L - self.seq_len)[:self.batch_size]:
            end_idx = start_idx+self.seq_len
            b_features[start_idx] = audio_features[0,start_idx:end_idx]
            b_frame_times[start_idx] = frame_times[start_idx:end_idx]

            left_idx = int(th.searchsorted(timestamps, frame_times[start_idx], right=False))
            right_idx = int(th.searchsorted(timestamps, frame_times[end_idx], right=True))
            b_ranges.append((left_idx, right_idx))
            max_tokens = max(max_tokens, right_idx - left_idx)

        b_tokens = th.full((self.batch_size, max_tokens+1), PAD)
        b_timestamps = th.full((self.batch_size, max_tokens+1), th.inf).float()

        for idx, (left_idx, right_idx) in enumerate(b_ranges):
            num_tokens = right_idx-left_idx
            b_tokens[idx, :num_tokens] = tokens[0,left_idx:right_idx]
            b_tokens[idx, num_tokens] = EOS
            b_timestamps[idx, :num_tokens] = timestamps[0,left_idx:right_idx]
            b_timestamps[idx, num_tokens] = th.inf

        return b_frame_times, b_features, b_timestamps, b_tokens


    def forward(
        self,
        labels: Float[Tensor, str(f"1 {NUM_LABELS}")],
        audio: Float[Tensor, str(f"1 {A_DIM} L")],
        tokens: Int[Tensor, "1 N"],
        timestamps: Float[Tensor, "1 N"],
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        
        b_frame_times, b_features, b_timestamps, b_tokens = self.make_batch(audio, tokens, timestamps)

        # randomly mask labels for training
        b_labels = labels.repeat(self.batch_size, 1)
        label_embs = self.label_emb(th.where(th.rand_like(b_labels) < .5, -1, b_labels))

        b_prelude_tokens = th.tensor([DIFF, BOS]).repeat(self.batch_size, 1)
        b_prelude_timestamps = th.zeros(self.batch_size, 2)

        h = self.decoder(
            x = self.embed(th.cat([b_prelude_tokens, b_tokens], dim=1)),
            x_t = th.cat([b_prelude_timestamps, b_timestamps], dim=1),
            ctx = b_features,
            ctx_t = b_frame_times,
            c = label_embs,
        ) # B N+2 E

        diff_emb, h = h[:,0], h[:,1:-1] # B E, B N E
        pred_labels = self.label_head(diff_emb) # B NUM_LABELS
        label_loss = (b_labels - pred_labels).pow(2).mean()

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

        loss = token_loss + timing_loss + label_loss
        return loss, {
            "loss": loss.detach(),
            "token": token_loss.detach(),
            "timing": timing_loss.detach(),
            "label": label_loss.detach(),
        }
    
    @th.no_grad
    def sample(
        self,
        audio: Float[Tensor, str(f"{A_DIM} L")],
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
        time_budget: float = float('inf'), # max allowed time (sec)
    ) -> tuple[
        list[list[tuple[int, float]]],          # list of B lists of (token, timestamp) tuples
        Float[Tensor, str(f"B {NUM_LABELS}")],  # predicted labels
    ]:
        
        end_time = time.time() + time_budget
        
        c = self.label_emb(labels) # B C
        pred_labels = []
        B = c.size(0)
        
        L = audio.size(-1)
        ctx = self.audio_encoder(F.pad(audio, (0, self.seq_len-1))[None]).transpose(1,2) # L+s-1 H
        ctx_t = th.tensor(get_frame_times(L+self.seq_len-1))[None].float() # L+s-1

        prelude_tokens = th.tensor([DIFF, BOS]).repeat(B,1)
        prelude_timestamps = th.full((B,2), 0)

        active_batches = th.arange(B)
        cur_i = th.zeros(B)
        cur_tail_idx = th.zeros(B)
        cur_tokens = th.empty(B,0)
        cur_timestamps = th.empty(B,0)

        output_tokens: list[list[tuple[int, float]]] = [ [] for _ in range(B) ]

        while True:
            if time.time() > end_time:
                # time limit reached
                break

            cur_ctx_idx = th.arange(self.seq_len)[None] + cur_i[:,None]
            cur_ctx = ctx[cur_ctx_idx] # B bL H
            cur_ctx_t = ctx_t[cur_ctx_idx] # B bL

            cur_x = self.embed(th.cat([prelude_tokens, cur_tokens], dim=1)) # B n+2 E
            cur_x_t = th.cat([prelude_timestamps, cur_timestamps], dim=1) # B n+2

            h = self.decoder( cur_x, cur_x_t, cur_ctx, cur_ctx_t, c ) # B n+2 E
            diff_emb, h = h[:,0], h[:,1:] # B E, B n+1 E
            cur_tail_emb = h[th.arange(B),cur_tail_idx] # B E

            pred_labels.append(self.label_head(diff_emb)) # B NUM_LABELS

            pred_token_logits = self.token_head(cur_tail_emb) # B V
            pred_timing_logits = self.timing_head(cur_tail_emb) # B s

            # disallow timing into the past
            cur_latest_offsets = cur_x_t[th.arange(B),cur_tail_idx] - ctx_t[cur_i] # B
            disallow_timing_mask = cur_latest_offsets[:,None] > th.arange(self.seq_len)[None]
            pred_timing_logits[disallow_timing_mask] = -th.inf

            pred_tokens = th.multinomial(pred_token_logits.softmax(dim=-1), num_samples=1)[:,0] # B
            pred_offsets = th.multinomial(pred_timing_logits.softmax(dim=-1), num_samples=1)[:,0] # B
            pred_timestamps = ctx_t[cur_i + pred_offsets] # B

            # update windows
            cur_i += th.where(
                pred_tokens == EOS,                                     # no tokens remaining for current window
                self.seq_len,                                           # - go to next window
                th.clamp(pred_offsets - int(self.seq_len * .5), min=0)  # - slide window forward until half of window is future context
            )

            # grow sequence
            if (cur_tail_idx >= cur_tokens.size(1)).any():
                cur_tokens = th.cat([cur_tokens, th.full((active_batches.size(0), 1), PAD)], dim=1)
                cur_timestamps = th.cat([cur_timestamps, th.full((active_batches.size(0), 1), th.inf)], dim=1)

            # update sequence
            pred_tokens[pred_tokens == EOS] = PAD
            cur_tokens[:, cur_tail_idx] = pred_tokens
            cur_timestamps[:, cur_tail_idx] = pred_timestamps
            cur_tail_idx += (pred_tokens != PAD).long()

            # dequeue tokens that are before start of new window
            shifts = th.searchsorted(cur_timestamps, ctx_t[cur_i,None])[:,0] # B
            cur_tokens = roll_by_shifts(cur_tokens, shifts)
            cur_timestamps = roll_by_shifts(cur_timestamps, shifts)
            cur_tail_idx -= shifts

            for b, (shift,o) in enumerate(zip(shifts, output_tokens)):
                sl = (b, slice(-shift,None)) # ...[b,-shift:]

                o.extend((
                    (int(token), float(timestamp)) 
                    for token, timestamp in zip(cur_tokens[sl], cur_timestamps[sl])
                ))
                cur_tokens[sl] = PAD
                cur_timestamps[sl] = th.inf

            # remove completed samples from batch 
            next_active = cur_i < L
            if not next_active.any():
                # all completed
                break

            prelude_tokens = prelude_tokens[next_active]
            prelude_timestamps = prelude_timestamps[next_active]
            active_batches = active_batches[next_active]
            cur_i = cur_i[next_active]
            cur_tail_idx = cur_tail_idx[next_active]
            cur_tokens = cur_tokens[next_active]
            cur_timestamps = cur_timestamps[next_active]

            # shrink sequence
            all_pad = (cur_tokens == PAD).all(dim=0) # N
            num_all_pad = th.argmax(th.cat([~all_pad.flip(dims=[0]), th.tensor([True])], dim=0))
            cur_tokens = cur_tokens[:,:num_all_pad]
            cur_timestamps = cur_timestamps[:,:num_all_pad]

        for b, (tail_idx,o) in enumerate(zip(cur_tail_idx, output_tokens)):
            sl = (b, slice(tail_idx)) # ...[b,:tail_idx]
            o.extend((
                (int(token), float(timestamp)) 
                for token, timestamp in zip(cur_tokens[sl], cur_timestamps[sl])
            ))

        return output_tokens, th.stack(pred_labels).mean(dim=0)


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

        true_label = batch[0][0]
        audio = batch[1][0]
        label = true_label.repeat(2,1)
        label = th.where(th.rand_like(label) < .5, -1, label)

        exp: SummaryWriter = self.logger.experiment # type: ignore

        def f_label(label):
            sr, ar, od, cs, hp = [ round(l, ndigits=1) for l in label ]
            return f'{sr=:>4} {ar=:>4} {od=:>4} {cs=:>4} {hp=:>4}'

        samples, pred_labels = self.sample(audio, label, time_budget=10)
        for i, (sample, pred_label) in enumerate(zip(samples, pred_labels)):
            sample_text = '\n'.join([
                f'true: {f_label(true_label)}',
                f'pred: {f_label(pred_label)}',
            ] + [
                f"{timestamp:0>10}: {str(decode(token))}"
                for token, timestamp in sample
            ])
            exp.add_text(f'sample/{i}', sample_text, global_step=self.global_step)