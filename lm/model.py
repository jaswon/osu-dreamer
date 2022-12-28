from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from x_transformers import ContinuousTransformerWrapper, Encoder, Decoder

from osu_dreamer.data import A_DIM
from osu_dreamer.tokens import FROM_IDX, VOCAB_SIZE, TIME, BOS, EOS, END


class Model(pl.LightningModule):
    def __init__(
        self,

        context_len: int,
        embed_dim: int,
        time_dim: int,
    
        learning_rate: float = 0.,
        learning_rate_schedule_factor: float = 0.,
        learning_rate_patience: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # model
        self.context_len = context_len
        self.audio_seq_length = 2**(time_dim-1)
        self.time_dim_mask = 2**torch.arange(time_dim)
        self.token_embeddings = nn.Embedding(VOCAB_SIZE, embed_dim)

        enc_audio_dim = 128

        self.enc = ContinuousTransformerWrapper(
            dim_in=A_DIM,
            max_seq_len=2**time_dim-1,
            attn_layers=Encoder(
                dim=enc_audio_dim,
                depth=6,
            ),
        ) # B,L,A -> B,L,D

        self.dec = ContinuousTransformerWrapper(
            dim_in=embed_dim+time_dim,
            dim_out=VOCAB_SIZE+time_dim,
            max_seq_len=context_len,
            use_abs_pos_emb = False,
            attn_layers=Decoder(
                dim=enc_audio_dim,
                depth=6,
                rotary_xpos = True,
                # rel_pos_bias = True,
                # use_rmsnorm = True,
                # sandwich_norm = True,
                cross_attend=True,
            ),
        ) # B,N,E+T + B,L,D -> B,N,V+T

        # training params

        self.learning_rate = learning_rate
        self.learning_rate_schedule_factor = learning_rate_schedule_factor
        self.learning_rate_patience = learning_rate_patience


    def to_time_embedding(self, times: "B,N") -> "B,N,T":
        mask = self.time_dim_mask.to(times.device, times.dtype)
        return times.unsqueeze(-1).bitwise_and(mask).ne(0).float()

    def from_time_embedding(self, times: "B,N,T") -> "B,N":
        mask = self.time_dim_mask.to(times.device, times.dtype)
        return torch.sum(mask * times.round(), -1).int()
        
    def forward(self, a: "B,A,L", tokens: "B,N", times: "B,N", mask: "B,N" = None):
        mask_args = dict(
            mask=mask.bool(),
            context_mask = torch.ones(z.shape[:-1], dtype=bool, device=z.device),
        ) if mask is not None else {}

        z: "B,L,D" = self.enc(a.permute(0,2,1))

        out: "B,N,V+T" = self.dec(torch.cat([
            self.token_embeddings(tokens),
            self.to_time_embedding(times),
        ], dim=2), context=z, **mask_args)
        return torch.tensor_split(out, (VOCAB_SIZE,), dim=-1)
    
    
#
#
# =============================================================================
# MODEL TRAINING
# =============================================================================
#
#

    def compute_loss(self, a: "B,A,L", mask: "B,N", tokens: "B,N", times: "B,N", true_tokens: "B,N", true_times: "B,N"):
        pred_tokens, pred_times = self(a, tokens, times, mask)

        classification_loss = F.cross_entropy(pred_tokens.flatten(0,1), true_tokens.flatten(0,1))

        time_idxs = tokens == TIME
        true_times: "B,N,T" = self.to_time_embedding(true_times)
        time_reconstruction_loss = 0
        if time_idxs.any():
            time_reconstruction_loss = F.binary_cross_entropy_with_logits(pred_times[time_idxs], true_times[time_idxs])

        return classification_loss + time_reconstruction_loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW([
            *self.enc.parameters(),
            *self.dec.parameters(),
            *self.token_embeddings.parameters(),
        ], lr=self.learning_rate)
        
        return dict(
            optimizer=opt,
            lr_scheduler=dict(
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, 
                    factor=self.learning_rate_schedule_factor,
                    patience=self.learning_rate_patience,
                ),
                monitor="val/loss",
            ),
        )
    
    def training_step(self, batch, *args, **kwargs):
        torch.cuda.empty_cache()
        loss = self.compute_loss(*batch)
        
        self.log(
            "train/loss", loss.detach(),
            logger=True, on_step=True, on_epoch=False,
        )
        
        return loss

    def validation_step(self, batch, *args, **kwargs):
        torch.cuda.empty_cache()
        loss = self.compute_loss(*batch)
        
        self.log(
            "val/loss", loss.detach(),
            logger=True, on_step=False, on_epoch=True,
        )

    def predict_step(self, a: "1,A,L", *args, **kwargs):
        assert a.size(0) == 1

        output = []
        num_lookback = 5

        recent_times = []
        cur = 0

        while True:
            print(f'{cur=}')
            a_seg: "1,A,l" = a[...,cur:cur+self.audio_seq_length]

            tokens: "1,1" = torch.tensor([[BOS]], device=a.device)
            times: "1,1" = torch.tensor([[-1]], device=a.device)
            cut: "1,1" = torch.tensor([0])

            while True:
                pred_tokens, pred_times = self(a_seg, tokens, times)

                next_token: "1,1" = pred_tokens[:,-1:].argmax(dim=-1)
                next_time: "1,1" = self.from_time_embedding(pred_times[:,-1:].sigmoid())

                next_token_i = next_token.item()
                print(f'{FROM_IDX[next_token_i]}')
                if next_token_i == EOS:
                    break
                elif next_token_i == TIME:
                    t = int(next_time.item())
                    output.append(t+cur)
                    recent_times.append(t)
                    cut += 1
                else:
                    if next_token_i == END and isinstance(output[-1], int):
                        # don't count spinner/slider ends in `recent_times`
                        recent_times.pop()
                        cut -= 1
                        cut[-1] = 0
                    output.append(FROM_IDX[next_token_i])

                tokens = torch.cat([tokens, next_token], dim=1)[:,-self.context_len:]
                times = torch.cat([times, next_time], dim=1)[:,-self.context_len:]
                cut = torch.cat([cut, torch.tensor([0])])[-self.context_len:]

            if cur+self.audio_seq_length >= a.size(-1):
                break

            if len(recent_times) > 0:
                cur_lookback = min(num_lookback, len(recent_times))
                cur += recent_times[-cur_lookback]
                cut_idx = torch.nonzero(cut == cur_lookback)[-1].item()
                tokens = tokens[:,cut_idx:]
                times = times[:,cut_idx:] - recent_times[-cur_lookback]
                cut = cut[cut_idx:]
                recent_times = recent_times[-cur_lookback:]
            else:
                # no objects placed yet
                cur += self.audio_seq_length

        return output