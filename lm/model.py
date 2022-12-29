from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

from einops import rearrange
from einops.layers.torch import Rearrange

import pytorch_lightning as pl

from x_transformers import ContinuousTransformerWrapper, Encoder, Decoder

from osu_dreamer.data import A_DIM
from osu_dreamer.tokens import FROM_IDX, VOCAB_SIZE, TIME, POSITION, BOS, EOS, END

from data import convert_img_to_pos

class PositionEncoder(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()

        self.net = nn.Sequential(*(
            nn.Conv2d(1 if i==0 else dim, dim, 4,2,1)
            for i in range(depth)
        )) # B,1,W,H -> B,dim,1,1

    def forward(self, x: '*,D,D') -> '*,C':
        *B, W, H = x.size()
        return self.net(x.reshape(-1, 1, W, H)).reshape(*B, W, H)

class PositionDecoder(nn.Sequential):
    def __init__(self, dim, depth):
        super().__init__()

        self.net = nn.Sequential(*(
            nn.ConvTranspose2d(dim, dim if i < depth-1 else 1, 4,2,1)
            for i in range(depth)
        )) # B,dim,1,1 -> B,1,W,H

    def forward(self, x: '*,C') -> '*,D,D':
        *B, D = x.size()
        return self.net(x.reshape(-1, D, 1, 1)).squeeze(1)


class Model(pl.LightningModule):
    def __init__(
        self,

        context_len: int,
        embed_dim: int,
        time_dim: int,
        pos_dim: int,
        pos_depth: int,
        h_dim: int,
        depth: int,

        topk: int,
    
        learning_rate: float = 0.,
        learning_rate_schedule_factor: float = 0.,
        learning_rate_patience: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # model
        self.topk = topk
        self.context_len = context_len
        self.audio_seq_length = 2**(time_dim-1)
        self.time_dim = time_dim
        self.time_dim_mask = 2**torch.arange(time_dim)
        self.token_embeddings = nn.Embedding(VOCAB_SIZE, embed_dim)

        self.pos_enc = PositionEncoder(pos_dim, pos_depth)
        self.pos_dec = PositionDecoder(pos_dim, pos_depth)

        self.enc = ContinuousTransformerWrapper(
            dim_in=A_DIM,
            max_seq_len=2**time_dim-1,
            attn_layers=Encoder(
                dim=h_dim,
                depth=depth,
            ),
        ) # B,L,A -> B,L,D

        self.dec = ContinuousTransformerWrapper(
            dim_in=embed_dim+time_dim+pos_dim,
            dim_out=VOCAB_SIZE+time_dim+pos_dim,
            max_seq_len=context_len,
            use_abs_pos_emb = False,
            attn_layers=Decoder(
                dim=h_dim,
                depth=depth,
                rotary_xpos = True,
                # rel_pos_bias = True,
                # use_rmsnorm = True,
                # sandwich_norm = True,
                cross_attend=True,
            ),
        ) # B,N,E+T+P + B,L,D -> B,N,V+T+P

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
        
    def forward(self, a: "B,A,L", tokens: "B,N", times: "B,N", positions: "B,N,D,D", mask: "B,N" = None):
        z: "B,L,D" = self.enc(a.permute(0,2,1))
        
        mask_args = dict(
            mask=mask.bool(),
            context_mask = torch.ones(z.shape[:-1], dtype=bool, device=z.device),
        ) if mask is not None else {}

        out: "B,N,V+T" = self.dec(torch.cat([
            self.token_embeddings(tokens),
            self.to_time_embedding(times),
            self.pos_enc(positions),
        ], dim=-1), context=z, **mask_args)

        return torch.tensor_split(out, (VOCAB_SIZE,VOCAB_SIZE+self.time_dim), dim=-1)
    
    
#
#
# =============================================================================
# MODEL TRAINING
# =============================================================================
#
#

    def compute_loss(self, a: "B,A,L", mask: "B,N", tokens: "B,N", times: "B,N", positions: "B,N,D,D", true_tokens: "B,N", true_times: "B,N", true_positions: "B,N,D,D", val=False):
        torch.cuda.empty_cache()

        pred_tokens, pred_time_embs, pred_pos_embs = self(a, tokens, times, positions, mask)

        # token loss
        token_loss = F.cross_entropy(pred_tokens.flatten(0,1), true_tokens.flatten(0,1))

        # position loss
        pos_idxs = tokens == POSITION
        pred_positions: 'B,N,D,D' = self.pos_dec(pred_pos_embs)
        position_loss = 0
        if pos_idxs.any():
            position_loss = F.smooth_l1_loss(pred_positions[time_idxs], true_positions[time_idxs])

        # time loss
        time_idxs = tokens == TIME
        true_time_embs: "B,N,T" = self.to_time_embedding(true_times)
        time_loss = 0
        if time_idxs.any():
            time_loss = F.binary_cross_entropy_with_logits(pred_time_embs[time_idxs], true_time_embs[time_idxs])

        loss = token_loss + time_loss + position_loss

        # log
        step = 'val' if val else 'train'
        self.log_dict({
            f"{step}/loss": loss.detach(),
            f"{step}/token_loss": token_loss.detach(),
            f"{step}/time_loss": time_loss.detach(),
            f"{step}/position_loss": position_loss.detach(),
        }, logger=True, on_step=not val, on_epoch=val)

        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW([
            *self.pos_enc.parameters(),
            *self.pos_dec.parameters(),
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
        return self.compute_loss(*batch)

    def validation_step(self, batch, *args, **kwargs):
        self.compute_loss(*batch, val=True)
        
    def predict_step(self, a: "1,A,L", *args, **kwargs):
        assert a.size(0) == 1

        output = []

        # `recent_times` is a list of most recent start times, up to a maximum of `num_lookback`
        # the first value will be used to start the next audio segment
        # `slice_idxs` will be used to reindex the token arrays for the next audio segment
        num_lookback = 5
        recent_times = []
        slice_idxs = []

        audio_idx = 0
        while True:
            print(f'{audio_idx=}')
            a_seg: "1,A,l" = a[...,audio_idx:audio_idx+self.audio_seq_length]

            tokens: "1,N" = torch.tensor([[BOS]], device=a.device)
            times: "1,N" = torch.tensor([[-1]], device=a.device)
            positions: "1,N,D,D" = torch.zeros(1,1,2**self.depth, 2**self.depth, device=a.device)

            while True:
                pred_tokens, pred_time_embs, pred_pos_embs = self(a_seg, tokens, times, positions)

                # topk_weights, topk_idxs = torch.topk(pred_tokens[0, -1], k=self.topk, dim=-1) 
                # next_token: "1,1" = topk_idxs[None, None, next(iter(WeightedRandomSampler(topk_weights, 1)))]
                next_token: "1,1" = pred_tokens[:,-1:].argmax(dim=-1)

                next_time: "1,1" = torch.tensor([[-1]], device=a.device)
                next_pos: "1,1,D,D" = torch.zeros(2**self.pos_depth, 2**self.pos_depth, device=a.device)

                next_token_i = next_token.item()
                print(f'{FROM_IDX[next_token_i]}')
                if next_token_i == EOS:
                    break
                elif next_token_i == START_TIME:
                    next_time: "1,1" = self.from_time_embedding(pred_time_embs[:,-1:].sigmoid())
                    t = int(next_time.item())
                    output.append(('START_TIME', t+audio_idx))

                    # update recent times
                    recent_times = (recent_times + [t])[-min(len(recent_times), num_lookback):]
                    slice_idxs.append(tokens.size(1))
                elif next_token_i == END_TIME:
                    next_time: "1,1" = self.from_time_embedding(pred_time_embs[:,-1:].sigmoid())
                    t = int(next_time.item())
                    output.append(('END_TIME', t+audio_idx))
                elif next_token_i == POSITION:
                    next_pos: "1,1,D,D" = self.pos_dec(pred_pos_embs[:,-1:])
                    output.append(('POSITION', *convert_img_to_pos(next_pos)))
                else:
                    output.append(FROM_IDX[next_token_i])

                tokens = torch.cat([tokens, next_token], dim=1)
                times = torch.cat([times, next_time], dim=1)
                positions = torch.cat([positions, next_pos], dim=1)

                if tokens.size(1) > self.context_len:
                    tokens = tokens[:,1:]
                    times = times[:,1:]
                    positions = positions[:,1:]
                    slice_idxs = [ i-1 for i in slice_idxs ]

            if audio_idx+self.audio_seq_length >= a.size(-1):
                break

            if len(recent_times) > 0 and recent_times[0] > 0:
                # at least one new object has been placed in this section
                offset, slice_idx = recent_times[0], slice_idxs[0]
                audio_idx += offset
                tokens = tokens[:,slice_idx:]
                times = times[:,slice_idx:] - offset
                positions = positions[:,slice_idx:]
                recent_times = [ t - offset for t in recent_times ]
                slice_idxs = [ i - slice_idx for i in slice_idxs ]
            else:
                # no objects placed in this section, start next section from scratch
                audio_idx += self.audio_seq_length
                tokens: "1,N" = torch.tensor([[BOS]], device=a.device)
                times: "1,N" = torch.tensor([[-1]], device=a.device)
                positions: "1,N,D,D" = torch.zeros(1,1,2**self.depth, 2**self.depth, device=a.device)


        return output
