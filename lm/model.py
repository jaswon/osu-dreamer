from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from x_transformers import ContinuousTransformerWrapper, TransformerWrapper, Encoder, Decoder

from osu_dreamer.data import A_DIM
from osu_dreamer.tokens import VOCAB_SIZE, TIME


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
        self.time_dim = time_dim
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
            # use_abs_pos_emb = False,
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
        return torch.sum(mask * times.round(), -1)
        
    def forward(self, a: "N,A,L", x: "N,X,L" = None):
        pass
    
    
#
#
# =============================================================================
# MODEL TRAINING
# =============================================================================
#
#

    def compute_loss(self, a: "B,A,L", toks: "B,N", times: "B,N", true_toks: "B,N", true_times: "B,N"):

        time_idxs = (toks == TIME).clone()

        toks: "B,N,E" = self.token_embeddings(toks)
        times: "B,N,T" = self.to_time_embedding(times)
        true_times: "B,N,T" = self.to_time_embedding(true_times)

        z: "B,L,D" = self.enc(a.permute(0,2,1))
        out: "B,N,V+T" = self.dec(torch.cat([toks, times],dim=2), context=z)
        out_toks, out_times = torch.tensor_split(out, (VOCAB_SIZE,), dim=-1)

        classification_loss = F.cross_entropy(out_toks.flatten(0,1), true_toks.flatten(0,1))

        time_reconstruction_loss = 0
        if time_idxs.any():
            time_reconstruction_loss = F.binary_cross_entropy_with_logits(out_times[time_idxs], true_times[time_idxs])

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