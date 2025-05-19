
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
    col_idxs = th.arange(n, device=shifts.device).repeat(b,1)
    shifted_idxs = (col_idxs + shifts[:,None]) % n
    return th.gather(input, 1, shifted_idxs.long())

    
class Model(pl.LightningModule):
    def __init__(
        self,

        # training parameters
        batch_size: int,
        seq_len: int,
        opt_args: dict[str, Any],
        focal_gamma: float,
        max_tokens: int,

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
        self.max_tokens = max_tokens

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
        self.label_head = nn.Linear(embed_dim, NUM_LABELS)

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
        D = audio.device
        L = audio.size(-1)
        audio_features = self.audio_encoder(audio).transpose(1,2)
        frame_times = th.tensor(get_frame_times(L), device=D).float()

        b_features = th.empty(self.batch_size, self.seq_len, audio_features.size(-1), device=D) # B H bL
        b_frame_times = th.empty(self.batch_size, self.seq_len, device=D) # B bL
        b_ranges: list[tuple[int,int]] = []
        max_tokens: int = 0

        idx = 0
        for start_idx in th.randperm(L - self.seq_len):
            if idx == self.batch_size:
                break
            end_idx = start_idx+self.seq_len
            left_idx = int(th.searchsorted(timestamps[0], frame_times[start_idx], right=False))
            right_idx = int(th.searchsorted(timestamps[0], frame_times[end_idx], right=True))
            num_tokens = right_idx - left_idx
            if num_tokens > self.max_tokens:
                continue

            b_features[idx] = audio_features[0,start_idx:end_idx]
            b_frame_times[idx] = frame_times[start_idx:end_idx]
            b_ranges.append((left_idx, right_idx))
            max_tokens = max(max_tokens, num_tokens)
            idx += 1

        b_tokens = th.full((self.batch_size, max_tokens+1), PAD, device=D)
        b_timestamps = th.full((self.batch_size, max_tokens+1), th.inf, device=D).float()

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
        
        D = audio.device
        b_frame_times, b_features, b_timestamps, b_tokens = self.make_batch(audio, tokens, timestamps)

        # randomly mask labels for training
        b_labels = labels.repeat(self.batch_size, 1)
        label_embs = self.label_emb(th.where(th.rand_like(b_labels) < .5, -1, b_labels))

        b_prelude_tokens = th.tensor([DIFF, BOS], device=D).repeat(self.batch_size, 1)
        b_prelude_timestamps = th.zeros(self.batch_size, 2, device=D)

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
            "src_len": th.tensor(b_tokens.size(1)+2).float(),
        }
    
    @th.no_grad
    def sample(
        self,
        audio: Float[Tensor, str(f"{A_DIM} L")],
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
        time_budget: int | float = float('inf'), # max allowed time (sec)
    ) -> tuple[
        list[list[tuple[int, float]]],          # list of B lists of (token, timestamp) tuples
        Float[Tensor, str(f"B {NUM_LABELS}")],  # predicted labels
    ]:
        D = audio.device
        end_time = time.time() + time_budget
        
        c = self.label_emb(labels) # B C
        B = c.size(0)
        pred_labels = [ [] for _ in range(B) ]
        
        L = audio.size(-1)
        ctx = self.audio_encoder(F.pad(audio, (0, self.seq_len-1))[None])[0].transpose(0,1) # L+s-1 H
        ctx_t = th.tensor(get_frame_times(L+self.seq_len-1), device=D).float() # L+s-1

        prelude_tokens = th.tensor([DIFF, BOS], device=D).long().repeat(B,1)
        prelude_timestamps = th.full((B,2), 0, device=D)

        active_batches = th.arange(B, device=D)
        cur_i = th.zeros(B, device=D).long()
        cur_tail_idx = th.zeros(B, device=D).long()
        cur_tokens = th.empty(B,0, device=D).long()
        cur_timestamps = th.empty(B,0, device=D)

        output_tokens: list[list[tuple[int, float]]] = [ [] for _ in range(B) ]

        while True:
            if time.time() > end_time:
                # time limit reached
                break

            cur_ctx_idx = th.arange(self.seq_len, device=D)[None] + cur_i[:,None]
            cur_ctx = ctx[cur_ctx_idx] # B bL H
            cur_ctx_t = ctx_t[cur_ctx_idx] # B bL

            cur_x = self.embed(th.cat([prelude_tokens, cur_tokens], dim=1)) # B n+2 E
            cur_x_t = th.cat([prelude_timestamps, cur_timestamps], dim=1) # B n+2

            h = self.decoder( cur_x, cur_x_t, cur_ctx, cur_ctx_t, c ) # B n+2 E
            diff_emb, h = h[:,0], h[:,1:] # B E, B n+1 E
            cur_tail_emb = h[th.arange(active_batches.size(0)),cur_tail_idx] # B E

            for i, pred_label in zip(active_batches, self.label_head(diff_emb)):
                pred_labels[i].append(pred_label)

            pred_token_logits = self.token_head(cur_tail_emb) # B V
            pred_timing_logits = self.timing_head(cur_tail_emb) # B s

            # disallow timing into the past
            cur_latest_timestamps = cur_x_t[th.arange(active_batches.size(0)),1+cur_tail_idx] # B
            disallow_timing_mask = cur_ctx_t < cur_latest_timestamps[:,None]
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
                cur_tokens = th.cat([cur_tokens, th.full((active_batches.size(0), 1), PAD, device=D)], dim=1)
                cur_timestamps = th.cat([cur_timestamps, th.full((active_batches.size(0), 1), th.inf, device=D)], dim=1)

            # update sequence
            pred_tokens[pred_tokens == EOS] = PAD
            cur_tokens[th.arange(active_batches.size(0)), cur_tail_idx] = pred_tokens
            cur_timestamps[th.arange(active_batches.size(0)), cur_tail_idx] = pred_timestamps
            cur_tail_idx += (pred_tokens != PAD).long()

            # dequeue tokens that are before start of new window
            shifts = th.searchsorted(cur_timestamps.contiguous(), ctx_t[cur_i,None])[:,0] # B
            cur_tokens = roll_by_shifts(cur_tokens, shifts)
            cur_timestamps = roll_by_shifts(cur_timestamps, shifts)
            cur_tail_idx -= shifts

            for b, (shift,batch_idx) in enumerate(zip(shifts, active_batches)):
                sl = (b, slice(cur_tokens.size(1)-shift,None)) # ...[b,-shift:]

                output_tokens[batch_idx].extend((
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

            if not next_active.all():
                prelude_tokens = prelude_tokens[next_active]
                prelude_timestamps = prelude_timestamps[next_active]
                active_batches = active_batches[next_active]
                cur_i = cur_i[next_active]
                cur_tail_idx = cur_tail_idx[next_active]
                cur_tokens = cur_tokens[next_active]
                cur_timestamps = cur_timestamps[next_active]
                c = c[next_active]

            # shrink sequence
            while (cur_tokens[:,-1] == PAD).all():
                cur_tokens = cur_tokens[:,:-1]
                cur_timestamps = cur_timestamps[:,:-1]

        for b, (tail_idx,batch_idx) in enumerate(zip(cur_tail_idx, active_batches)):
            sl = (b, slice(tail_idx)) # ...[b,:tail_idx]
            output_tokens[batch_idx].extend((
                (int(token), float(timestamp)) 
                for token, timestamp in zip(cur_tokens[sl], cur_timestamps[sl])
            ))

        return output_tokens, th.stack([ th.stack(l).mean(dim=0) for l in pred_labels ])


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

        def f_label(label: Float[Tensor, str(f"{NUM_LABELS}")]):
            sr, ar, od, cs, hp = [ round(l.item(), ndigits=1) for l in label ]
            return f'{sr=:>4} {ar=:>4} {od=:>4} {cs=:>4} {hp=:>4}'

        samples, pred_labels = self.sample(audio, label, time_budget=10)
        for i, (sample, pred_label) in enumerate(zip(samples, pred_labels)):
            sample_text = '\n'.join([
                f'true: {f_label(true_label)}',
                f'pred: {f_label(pred_label)}',
                '',
                '|timestamp|token|',
                '|-|-|',
            ] + [
                f"|{round(timestamp/1000, ndigits=2)}|{str(decode(token))}|"
                for token, timestamp in sample
            ])
            exp.add_text(f'sample/{i}', sample_text, global_step=self.global_step)