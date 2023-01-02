from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import WeightedRandomSampler

import pytorch_lightning as pl

from x_transformers import ContinuousTransformerWrapper, Encoder, Decoder

from osu_dreamer.data import A_DIM
from osu_dreamer.tokens import FROM_IDX, VOCAB_SIZE, POSITION, BOS, EOS, START_TIME, END_TIME


def B(alpha, beta):
    return torch.exp(torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta))

### TIME

class TimeEmbedding(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()

        self.dim = dim
        self.size = 2**depth

        self.line: '1,L,1' = torch.arange(2**depth, dtype=float)[None,:,None]

        self.encoder = nn.Sequential(*(
            layer for i in range(depth)
            for layer in [
                nn.Identity() if i==0 else nn.GroupNorm(1,dim),
                nn.GELU(),
                nn.Conv1d(1 if i==0 else dim, dim, 4,2,1),
            ]
        )) # B,1,L -> B,dim,1

        self.decoder = nn.Sequential(*(
            layer for i in range(depth)
            for layer in [
                nn.GroupNorm(1,dim),
                nn.GELU(),
                nn.ConvTranspose1d(dim, dim if i < depth-1 else 1, 4,2,1),
            ]
        ), nn.Sigmoid()) # B,dim,1 -> B,1,L

    def time_to_class(self, times: '*,1') -> '*,':
        """convert frame indices to class indices"""
        return times.round().long().squeeze(-1)

    def signal_from_time(self, times: '*,1', sigma: float = 5.) -> "*,L":
        """
        returns a signal encoding a time as the mean of a 1D Gaussian

        if `times[i]` contains nan then `return[i]` will contain all zeros
        """
        diff: 'N,L,1' = self.line.to(times.device) - times.reshape(-1,1,1)
        pdf: 'N,L' = torch.exp(-.5 * (diff ** 2).sum(dim=-1) / sigma ** 2 )
        # pdf: 'N,L' = torch.exp(torch.lgamma())
        return torch.nan_to_num(pdf.reshape(*times.size()[:-1], *self.line.shape[1:-1]), nan=0)

    def to_time(self, embs: '*,C') -> '*,1':
        """
        returns the time encoded by the return value of `from_signal`
        """

        signal = self.to_signal(embs)
        *B,L = signal.size()

        signal = signal.reshape(-1,L)
        # signal /= signal.sum(dim=1)

        signal_max: 'N,' = signal.argmax(dim=-1)

        return self.line.to(signal.device)[0,signal_max].reshape(*B,1).round().int()

        times = (self.line.to(signal.device) * signal[...,None]).sum(dim=1)

        return times.reshape(*B,1).round().int()

    def from_signal(self, x: '*,L') -> '*,C':
        *B, L = x.size()
        return self.encoder(x.reshape(-1, 1, L)).reshape(*B, self.dim)

    def to_signal(self, x: '*,C') -> '*,L':
        *B, C = x.size()
        return self.decoder(x.reshape(-1, C, 1)).reshape(*B, *self.line.shape[1:-1])


### POSITON

class PositionEmbedding(nn.Module):
    def __init__(self, dim, depth):
        super().__init__()

        self.dim = dim
        self.depth = depth

        self.grid: '1,D,D,2' = torch.stack(torch.meshgrid(
            torch.linspace(-256,512+256,2**depth),
            torch.linspace(-192,384+192,2**depth),
            indexing='ij',
        ), dim=-1)[None]

        self.encoder = nn.Sequential(*(
            layer for i in range(depth)
            for layer in [
                nn.GELU(),
                nn.Conv2d(1 if i==0 else dim, dim, 4,2,1),
            ]
        )) # B,1,W,H -> B,dim,1,1

        self.decoder = nn.Sequential(*(
            layer for i in range(depth)
            for layer in [
                nn.GELU(),
                nn.ConvTranspose2d(dim, dim if i < depth-1 else 1, 4,2,1),
            ]
        ), nn.Sigmoid()) # B,dim,1,1 -> B,1,W,H

    def image_from_position(self, positions: '*,2', sigma: float = 5.) -> "*,D,D":
        """
        returns a square matrix encoding a position on the osu! playfield as the mean of a 2D Gaussian

        if `position[i]` contains nan then `return[i]` will contain all zeros
        """

        diff: 'N,D,D,2' = self.grid.to(positions.device) - positions.reshape(-1, 1, 1, 2)
        pdf = torch.exp(-.5 * (diff ** 2).sum(dim=-1) / sigma ** 2)
        return torch.nan_to_num(pdf.reshape(*positions.size()[:-1], *self.grid.shape[1:-1]), nan=0)

    def to_position(self, embs: '*,C') -> '*,2':
        """
        returns the position represented by the return value of `to_image`
        """

        image = self.to_image(embs)
        *B,W,H = image.size()

        image = image.reshape(-1,W,H)
        image /= image.sum(dim=(1,2))

        # image_max: 'B,' = image.flatten(-2).argmax(dim=-1)

        # return self.grid.to(image.device).flatten(1,2)[0,image_max].reshape(*B,2).round().int()

        positions = (self.grid.to(image.device) * image[...,None]).sum(dim=(1,2))

        return positions.reshape(*B,2).floor().int()

    def from_image(self, x: '*,D,D') -> '*,C':
        *B, W, H = x.size()
        return self.encoder(x.reshape(-1, 1, W, H)).reshape(*B, self.dim)

    def to_image(self, x: '*,C') -> '*,D,D':
        *B, C = x.size()
        return self.decoder(x.reshape(-1, C, 1, 1)).reshape(*B,*self.grid.shape[1:-1])


class Model(pl.LightningModule):
    def __init__(
        self,

        context_len: int,
        token_dim: int,

        time_dim: int,
        time_depth: int,

        pos_dim: int,
        pos_depth: int,

        h_dim: int,
        attn_depth: int,

        topk: int,

        learning_rate: float = 0.,
        learning_rate_schedule_factor: float = 0.,
        learning_rate_patience: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # training params
        self.learning_rate = learning_rate
        self.learning_rate_schedule_factor = learning_rate_schedule_factor
        self.learning_rate_patience = learning_rate_patience

        # model
        self.topk = topk
        self.context_len = context_len
        self.audio_seq_length = 2**time_depth

        self.token_emb = nn.Embedding(VOCAB_SIZE, token_dim)
        self.time_emb = TimeEmbedding(time_dim, time_depth)
        self.pos_emb = PositionEmbedding(pos_dim, pos_depth)

        self.enc = ContinuousTransformerWrapper(
            dim_in=A_DIM,
            max_seq_len=self.audio_seq_length,
            attn_layers=Encoder(
                dim=h_dim,
                depth=attn_depth,
                rel_pos_bias=True,
                ff_glu = True,
                ff_no_bias = True,
                use_rmsnorm = True,
            ),
        ) # B,L,A -> B,L,D

        *self.dec_split, dim_out = torch.cumsum(torch.tensor([VOCAB_SIZE,time_dim,pos_dim]), dim=0)
        self.dec = ContinuousTransformerWrapper(
            dim_in=token_dim+time_dim+pos_dim,
            dim_out=dim_out,
            max_seq_len=context_len,
            use_abs_pos_emb=False,
            attn_layers=Decoder(
                dim=h_dim,
                depth=attn_depth,
                deepnorm = True,
                ff_glu = True,
                ff_no_bias = True,
                sandwich_coef = 3,
                attn_sparse_topk = 8,
                attn_one_kv_head = True,
                rotary_xpos = True,
                rel_pos_bias = True,
                use_rmsnorm = True,
                # sandwich_norm = True,
                cross_attend=True,
                dec_cross_residual_attn = True,
            ),
        ) # B,N,E+T+P + B,L,D -> B,N,V+T+P

    def forward(self, a: "B,A,L", tokens: "B,N", times: "B,N,1", positions: "B,N,2", mask: "B,N" = None):
        z: "B,L,D" = self.enc(a.permute(0,2,1))

        mask_args = dict(
            mask=mask.bool(),
            context_mask = torch.ones(z.shape[:-1], dtype=bool, device=z.device),
        ) if mask is not None else {}

        time_signals: 'B,N,L' = self.time_emb.signal_from_time(times).type_as(a)
        position_imgs: 'B,N,D,D' = self.pos_emb.image_from_position(positions).type_as(a)

        out: "B,N,V+T+P" = self.dec(torch.cat([
            self.token_emb(tokens),
            self.time_emb.from_signal(time_signals),
            self.pos_emb.from_image(position_imgs),
        ], dim=-1), context=z, **mask_args)

        return torch.tensor_split(out, self.dec_split, dim=-1)


#
#
# =============================================================================
# MODEL TRAINING
# =============================================================================
#
#

    def compute_loss(self, batch, val=False, log_pred=False):
        torch.cuda.empty_cache()
        (
            a, # B,A,L
            mask, # B,N
            tokens, # B,N
            times, # B,N,1
            positions, # B,N,2
            true_tokens, # B,N
            true_times, # B,N,1
            true_positions, # B,N,2
        ) = batch

        pred_tokens, pred_time_embs, pred_pos_embs = self(a, tokens, times, positions, mask)

        if log_pred:
            true_toks = []
            for i, token in enumerate(tokens[0]):
                token = token.item()
                if token == POSITION:
                    x,y = [ round(i) for i in positions[0,i].tolist() ]
                    true_toks.append(f'({x},{y})')
                elif token == START_TIME:

                    # sig = self.time_emb.signal_from_time(times[0,i]).cpu().numpy()

                    # import matplotlib.pyplot as plt
                    # fig,ax = plt.subplots()
                    # ax.plot(self.time_emb.line[0,:,0], sig)
                    # ax.set_title(str(times[0,i]))

                    # fig.tight_layout()
                    # fig.savefig('true.png')
                    # plt.close(fig)

                    t = round(times[0,i].item())
                    true_toks.append(f'START: {t}')
                elif token == END_TIME:
                    t = round(times[0,i].item())
                    true_toks.append(f'END: {t}')
                else:
                    true_toks.append(FROM_IDX[token])


            import matplotlib.pyplot as plt
            sig_rem=10
            fig,ax = plt.subplots(nrows=sig_rem, figsize=(5,20))

            pred_toks = []
            for i,token in enumerate(pred_tokens[0]):
                token = token.argmax(dim=-1).item()
                if token == POSITION:
                    x,y = [ round(i.item()) for i in self.pos_emb.to_position(pred_pos_embs[0,i]) ]
                    pred_toks.append(f'({x},{y})')
                elif token == START_TIME:
                    t = round(self.time_emb.to_time(pred_time_embs[0,i]).item())

                    if sig_rem > 0:
                        sig = F.softmax(self.time_emb.to_signal(pred_time_embs[0,i]), dim=-1).detach().cpu().numpy()
                        ax[sig_rem-1].plot(self.time_emb.line[0,:,0], sig)
                        sig_rem -= 1

                    pred_toks.append(f'START: {t}')
                elif token == END_TIME:
                    t = round(self.time_emb.to_time(pred_time_embs[0,i]).item())
                    pred_toks.append(f'END: {t}')
                else:
                    pred_toks.append(FROM_IDX[token])

            fig.tight_layout()
            fig.savefig('pred.png')
            plt.close(fig)


            for true_tok, pred_tok,_ in zip(true_toks, pred_toks, range(50)):
                print(f'{str(true_tok):<15}{pred_tok}')

        # token loss
        token_loss = F.cross_entropy(pred_tokens.flatten(0,1), true_tokens.flatten(0,1))

        # position loss
        pos_idxs: 'B,N' = tokens == POSITION
        pred_position_imgs: 'B,N,D,D' = self.pos_emb.to_image(pred_pos_embs)
        true_position_imgs: 'B,N,D,D' = self.pos_emb.image_from_position(true_positions)
        position_loss = torch.tensor(0.)
        if pos_idxs.any():
            position_loss = F.mse_loss(pred_position_imgs[pos_idxs].float(), true_position_imgs[pos_idxs].float())

        # time loss
        time_idxs: 'B,N' = (tokens == START_TIME) | (tokens == END_TIME)
        pred_time_signals: 'B,N,L' = self.time_emb.to_signal(pred_time_embs)
        true_time_signals: 'B,N,1' = self.time_emb.time_to_class(true_times)
        time_loss = torch.tensor(0.)
        if time_idxs.any():
            print(pred_time_signals[time_idxs].shape)
            print(true_time_signals[time_idxs].max())
            print(true_time_signals[time_idxs].min())
            time_loss = F.cross_entropy(pred_time_signals[time_idxs], true_time_signals[time_idxs])

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
            *self.time_emb.parameters(),
            *self.pos_emb.parameters(),
            *self.enc.parameters(),
            *self.dec.parameters(),
            *self.token_emb.parameters(),
        ], lr=self.learning_rate)
        
        return dict(
            optimizer=opt,
            lr_scheduler=dict(
                interval='step',
                frequency=100,
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, 
                    factor=self.learning_rate_schedule_factor,
                    patience=self.learning_rate_patience,
                ),
                monitor="train/loss",
            ),
        )
    
    def training_step(self, batch, batch_idx, *args, **kwargs):
        return self.compute_loss(batch, log_pred=batch_idx % 10 == 0)

    def validation_step(self, batch, batch_idx, *args, **kwargs):
        self.compute_loss(batch, val=True)
        
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
            times: "1,N,1" = torch.full((1,1,1), torch.nan, device=a.device)
            positions: "1,N,2" = torch.full((1,1,2), torch.nan, device=a.device)

            while True:
                pred_tokens, pred_time_embs, pred_pos_embs = self(a_seg, tokens, times, positions)

                # topk_weights, topk_idxs = torch.topk(pred_tokens[0, -1], k=self.topk, dim=-1) 
                # next_token: "1,1" = topk_idxs[None, None, next(iter(WeightedRandomSampler(topk_weights, 1)))]
                next_token: "1,1" = pred_tokens[:,-1:].argmax(dim=-1)

                next_time: "1,1,1" = torch.full((1,1,1), torch.nan, device=a.device)
                next_pos: "1,1,2" = torch.full((1,1,2), torch.nan, device=a.device)

                next_token_i = next_token.item()
                if next_token_i == EOS:
                    print('EOS')
                    break
                elif next_token_i == START_TIME:
                    next_time: '1,1,1' = self.time_emb.to_time(pred_time_embs[:,-1:])
                    t = int(next_time.item())
                    output.append(('START_TIME', t+audio_idx))
                    print(('START_TIME', t+audio_idx))

                    # update recent times
                    recent_times = (recent_times + [t])[-min(len(recent_times), num_lookback):]
                    slice_idxs.append(tokens.size(1))
                elif next_token_i == END_TIME:
                    next_time: '1,1,1' = self.time_emb.to_time(pred_time_embs[:,-1:])
                    t = int(next_time.item())
                    output.append(('END_TIME', t+audio_idx))
                    print(('END_TIME', t+audio_idx))
                elif next_token_i == POSITION:
                    next_pos: '1,1,2'=  self.pos_emb.to_position(pred_pos_embs[:,-1:])
                    output.append(('POSITION', *next_pos[0,0].tolist()))
                    print(('POSITION', *next_pos[0,0].tolist()))
                else:
                    print(f'{FROM_IDX[next_token_i]}')
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
                times: "1,N,1" = torch.full((1,1,1), torch.nan, device=a.device)
                positions: "1,N,2" = torch.full((1,1,2), torch.nan, device=a.device)


        return output
