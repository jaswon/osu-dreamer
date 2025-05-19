
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
from .data.events import Event, EventType, encode, decode, PAD, BOS, EOS, VOCAB_SIZE, DIFF, T0

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
        
        try:
            encode(Event(EventType.TIMESTAMP, self.seq_len-1))
        except KeyError:
            raise ValueError('not enough timestamp tokens for sequence length- update `data/events.py`')

        # model
        audio_encoder = AudioEncoder.load_from_checkpoint(audio_encoder_ckpt)
        self.audio_encoder = audio_encoder.proj_audio
        self.a_dim = audio_encoder.a_dim

        self.embed = MP.Embedding(VOCAB_SIZE, embed_dim)
        token_head = MP.Linear(embed_dim, VOCAB_SIZE)
        token_head.weight = self.embed.weight
        self.token_head = nn.Sequential(
            token_head,
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

    @th.no_grad
    def make_batch(
        self,
        L: int,
        tokens: Int[Tensor, "N"],
        timestamps: Float[Tensor, "N"],
    ) -> tuple[
        Int[Tensor, "B S"],         # audio positioning 
        Int[Tensor, "B T"],         # token positioning
        Int[Tensor, "B T"],         # tokens
    ]:
        D = tokens.device
        frame_times = th.tensor(get_frame_times(L), device=D).float() # L

        b_feature_idxs = th.empty(self.batch_size, self.seq_len, device=D, dtype=th.int) # B S
        b_tokens = th.full((self.batch_size, self.max_tokens+1), PAD, device=D)
        b_token_idxs = th.full((self.batch_size, self.max_tokens+1), -1, device=D, dtype=th.int)

        timestamp_frame_idxs = th.argmin(abs(timestamps[:,None] - frame_times[None,:]), dim=1) # N

        idx = 0
        for start_idx in th.randperm(L - self.seq_len).tolist():
            if idx == self.batch_size:
                break
            end_idx = start_idx+self.seq_len
            left_idx, right_idx = th.searchsorted(
                timestamp_frame_idxs, 
                th.tensor([start_idx, end_idx], device=D),
            ).tolist()
            num_tokens = right_idx - left_idx
            if num_tokens > self.max_tokens:
                continue

            b_feature_idxs[idx] = th.arange(start_idx,end_idx)

            b_token_idxs[idx, :num_tokens] = timestamp_frame_idxs[left_idx:right_idx]
            assert ( True
                and (timestamp_frame_idxs[left_idx:right_idx] >= start_idx).all()
                and (timestamp_frame_idxs[left_idx:right_idx] <  end_idx).all()
            )
            b_tokens[idx, :num_tokens] = th.where(
                tokens[left_idx:right_idx] == -1,
                T0 + b_token_idxs[idx, :num_tokens] - start_idx,
                tokens[left_idx:right_idx],
            )
            b_tokens[idx, num_tokens] = EOS

            idx += 1

        return b_feature_idxs, b_token_idxs, b_tokens


    def forward(
        self,
        labels: Float[Tensor, str(f"1 {NUM_LABELS}")],
        audio: Float[Tensor, str(f"1 {A_DIM} L")],
        tokens: Int[Tensor, "1 N"],
        timestamps: Float[Tensor, "1 N"],
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        
        D = audio.device
        b_features = th.empty(self.batch_size, self.seq_len, self.a_dim)
        b_feature_idxs, b_token_idxs, b_tokens = self.make_batch(audio.size(-1), tokens[0], timestamps[0])
        b_features = self.audio_encoder(audio)[0].transpose(0,1)[b_feature_idxs,:]

        # randomly mask labels for training
        b_labels = labels.repeat(self.batch_size, 1)
        label_embs = self.label_emb(th.where(th.rand_like(b_labels) < .5, -1, b_labels))

        b_prelude_tokens = th.tensor([DIFF, BOS], device=D).repeat(self.batch_size, 1)
        b_prelude_idxs = th.full((self.batch_size,2), th.inf, device=D)

        b_prelude_idxs = b_feature_idxs[:,:1].repeat(1, 2)
        h = self.decoder(
            x = self.embed(th.cat([b_prelude_tokens, b_tokens], dim=1)),
            x_t = th.cat([b_prelude_idxs, b_token_idxs], dim=1),
            ctx = b_features,
            ctx_t = b_feature_idxs,
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

        loss = token_loss + label_loss
        return loss, {
            "loss": loss.detach(),
            "token": token_loss.detach(),
            "label": label_loss.detach(),
        }
    
    @th.no_grad
    def sample(
        self,
        audio: Float[Tensor, str(f"{A_DIM} L")],
        labels: Float[Tensor, str(f"B {NUM_LABELS}")],
        time_budget: int | float = float('inf'), # max allowed time (sec)
    ) -> tuple[
        list[list[Event|float]],                # list of B lists of events and timestamps
        Float[Tensor, str(f"B {NUM_LABELS}")],  # predicted labels
    ]:
        D = audio.device
        end_time = time.time() + time_budget
        
        c = self.label_emb(labels) # B C
        B = c.size(0)
        pred_labels = [ [] for _ in range(B) ]
        
        L = audio.size(-1)
        half_window = int(self.seq_len * .5)
        ctx = self.audio_encoder(F.pad(audio, (0, self.seq_len-1))[None])[0].transpose(0,1) # L+s-1 H
        frame_times = th.tensor(get_frame_times(L+self.seq_len-1))

        prelude_tokens = th.tensor([DIFF, BOS], device=D).long().repeat(B,1)

        active_batches = th.arange(B, device=D)             # samples that are still generating
        cur_start_idxs = th.zeros(B, device=D).long()       # current context window start index
        cur_tail_idx = th.zeros(B, device=D).long()         # token index @ end of generation
        cur_tokens = th.empty(B,0, device=D).long()         # tokens currently in generation
        cur_token_idxs = th.empty(B,0, device=D).long()     # token positioning

        output_tokens: list[list[Event | float]] = [ [] for _ in range(B) ]

        while True:
            if time.time() > end_time:
                # time limit reached
                break

            cur_ctx_idxs = th.arange(self.seq_len, device=D)[None] + cur_start_idxs[:,None]
            cur_ctx = ctx[cur_ctx_idxs] # B bL H

            prelude_token_idxs = cur_ctx_idxs[:,:1].repeat(1, 2)
            h = self.decoder(
                x = self.embed(th.cat([prelude_tokens, cur_tokens], dim=1)),
                x_t = th.cat([prelude_token_idxs, cur_token_idxs], dim=1),
                ctx = cur_ctx, 
                ctx_t = cur_ctx_idxs, 
                c = c,
            ) # B n+2 E

            # predict labels
            label_emb = h[:,0] # B E
            for i, pred_label in zip(active_batches, self.label_head(label_emb)):
                pred_labels[i].append(pred_label)

            # predict tokens
            pred_embs = h[:,1:] # B n+1 E
            cur_tail_emb = pred_embs[th.arange(active_batches.size(0)),cur_tail_idx] # B E
            pred_token_logits = self.token_head(cur_tail_emb) # B V

            # latest timing generated so far (no timing yet = ctx start)
            cur_latest_token_idxs = th.cat([
                cur_start_idxs[:,None], 
                cur_token_idxs,
            ], dim=1)[
                th.arange(active_batches.size(0)),
                cur_tail_idx,
            ] # B

            # disallow timing into the past
            for b_idx, num_mask in enumerate(cur_latest_token_idxs - cur_start_idxs):
                pred_token_logits[b_idx, T0:T0+num_mask] = -th.inf

            pred_tokens = th.multinomial(pred_token_logits.softmax(dim=-1), num_samples=1)[:,0] # B
            pred_token_idxs = th.where(
                pred_tokens >= T0,                      # timing predicted
                cur_start_idxs + pred_tokens - T0,      # ? use index from predicted timing
                cur_latest_token_idxs,                  # : use index from previously latest timing
            )

            # update windows
            cur_start_idxs += th.where(
                pred_tokens == EOS,                                         # no tokens remaining for current window
                self.seq_len,                                               # ? go to next window
                th.where(
                    pred_tokens >= T0,                                      # new timestamp predicted
                    th.clamp((pred_tokens - T0) - half_window, min=0),      # ? slide forward until half of window is future context
                    0,                                                      # : no window updates
                ),
            )

            # grow sequence
            if (cur_tail_idx >= cur_tokens.size(1)).any():
                cur_tokens = th.cat([cur_tokens, th.full((active_batches.size(0), 1), PAD, device=D)], dim=1)
                cur_token_idxs = th.cat([cur_token_idxs, th.full((active_batches.size(0), 1), -1, device=D)], dim=1)

            # update sequence
            pred_tokens[pred_tokens == EOS] = PAD
            cur_tokens[th.arange(active_batches.size(0)), cur_tail_idx] = pred_tokens
            cur_token_idxs[th.arange(active_batches.size(0)), cur_tail_idx] = pred_token_idxs
            cur_tail_idx += (pred_tokens != PAD).long()

            # dequeue tokens that are before start of new window
            shifts = (cur_token_idxs >= cur_start_idxs[:,None]).long().argmax(dim=1) # B
            cur_tokens = roll_by_shifts(cur_tokens, shifts)
            cur_token_idxs = roll_by_shifts(cur_token_idxs, shifts)
            cur_tail_idx -= shifts

            for b, (shift,batch_idx) in enumerate(zip(shifts, active_batches)):
                sl = (b, slice(cur_tokens.size(1)-shift,None)) # ...[b,-shift:]

                output_tokens[batch_idx].extend((
                    decode(int(token)) if token < T0 else float(frame_times[token_idx])
                    for token, token_idx in zip(cur_tokens[sl], cur_token_idxs[sl])
                ))
                cur_tokens[sl] = PAD
                cur_token_idxs[sl] = -1

            # remove completed samples from batch 
            next_active = cur_start_idxs < L
            if not next_active.any():
                # all completed
                break

            if not next_active.all():
                prelude_tokens = prelude_tokens[next_active]
                active_batches = active_batches[next_active]
                cur_start_idxs = cur_start_idxs[next_active]
                cur_tail_idx = cur_tail_idx[next_active]
                cur_tokens = cur_tokens[next_active]
                cur_token_idxs = cur_token_idxs[next_active]
                c = c[next_active]

            # shrink sequence
            while cur_tokens.size(1) > 0 and (cur_tokens[:,-1] == PAD).all():
                cur_tokens = cur_tokens[:,:-1]
                cur_token_idxs = cur_token_idxs[:,:-1]

        for b, (tail_idx,batch_idx) in enumerate(zip(cur_tail_idx, active_batches)):
            sl = (b, slice(tail_idx)) # ...[b,:tail_idx]
            output_tokens[batch_idx].extend((
                decode(int(token)) if token < T0 else float(frame_times[token_idx])
                for token, token_idx in zip(cur_tokens[sl], cur_token_idxs[sl])
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
            ] + [ str(event) for event in sample ])
            exp.add_text(f'sample/{i}', sample_text, global_step=self.global_step)