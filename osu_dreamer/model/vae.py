from typing import List

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    USE_MATPLOTLIB = True
except:
    USE_MATPLOTLIB = False

import numpy as np

import pytorch_lightning as pl

from osu_dreamer.data import A_DIM
from osu_dreamer.signal import (
    MAP_SIGNAL_DIM as X_DIM,
    TIMING_DIM as T_DIM,
)

from .vq import VectorQuantizer
from .modules import ConvNextBlock, Downsample, Upsample
    
class Encoder(nn.Sequential):
    def __init__(
        self,
        dims: List[int],
        convnext_mult: int,
    ):
        super().__init__(*(
            layer for ind, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:]))
            for layer in [
                ConvNextBlock(dim_in, dim_out, mult=convnext_mult),
                ConvNextBlock(dim_out, dim_out, mult=convnext_mult),
                Downsample(dim_out) if ind < (len(dims) - 2) else nn.Identity(),
            ]
        ))

        
class Decoder(nn.Sequential):
    def __init__(
        self,
        dims: List[int],
        convnext_mult: int,
        final_layer = nn.Tanh,
    ):
        super().__init__(*(
            layer for ind, (dim_in, dim_out) in enumerate(zip(dims[-1:0:-1], dims[-2::-1]))
            for layer in [
                ConvNextBlock(dim_in, dim_out, mult=convnext_mult),
                ConvNextBlock(dim_out, dim_out, mult=convnext_mult),
                Upsample(dim_out) if ind < (len(dims) - 2) else nn.Identity(),
            ]
        ), final_layer())
    
#
#
# ====================================
# MODEL
# ====================================
#
#

VALID_PAD = 1024

class Model(pl.LightningModule):
    def __init__(
        self,
        
        h_dims: List[int],
        convnext_mult: int,
        n_embed: int,
        
        timing_dropout: float,
        learning_rate: float = 0.,
        learning_rate_schedule_factor: float = 0.,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.timing_dropout = timing_dropout
        self.learning_rate = learning_rate
        self.learning_rate_schedule_factor = learning_rate_schedule_factor
        
        # models
        
        self.x_enc = Encoder([X_DIM, *h_dims], convnext_mult)
        self.a_enc = Encoder([A_DIM+T_DIM, *h_dims], convnext_mult)
        self.dec = Decoder([h_dims[:-1], h_dims[-1]*2], convnext_mult)
        self.vq  = VectorQuantizer(n_embed, h_dims[-1]*2)
        
    def inference_pad(self, x):
        x = F.pad(x, (VALID_PAD, VALID_PAD), mode='replicate')
        pad = (1 + x.size(-1) // 2 ** self.depth) * 2 ** self.depth - x.size(-1)
        x = F.pad(x, (0, pad), mode='replicate')
        return x, (..., slice(VALID_PAD,-(VALID_PAD+pad)))
    
    def decode(self, z: "N,H,L") -> "N,X,L":
        return self.dec(self.vq(z)[0])

    def forward(self, x, a, t):
        """x_hat"""
        z_x = self.x_enc(x)
        z_a = self.a_enc(torch.cat([a,t], dim=1))
        z = torch.cat([z_x, z_a], dim=1)
        return self.decode(z)
    
    def compute_losses(self, a: "N,A,L", t: "N,T,L", p: "N,T,L", x: "N,X,L", timing_dropout=0., pad=False):
        if pad:
            a, _ = self.inference_pad(a)
            t, _ = self.inference_pad(t)
            p, _ = self.inference_pad(p)
            x, _ = self.inference_pad(x)
            
        if timing_dropout > 0:
            drop_idxs = torch.randperm(t.size(0))[:int(t.size(0) * timing_dropout)]
            t[drop_idxs] = p[drop_idxs]

        # encode
        z_x = self.x_enc(x)
        z_a = self.a_enc(torch.cat([a,t], dim=1))
        z = torch.cat([z_x, z_a], dim=1)

        # quantize
        z_q, emb_loss, _ = self.vq(z)

        # decode
        x_hat = self.dec(z_q)
        
        # reconstruction loss
        rec = F.mse_loss(x_hat, x)

        return emb_loss, rec
     
#
#
# ====================================
# MODEL TRAINING
# ====================================
#
#

    def configure_optimizers(self):
        opt = torch.optim.AdamW([
            *self.x_enc.parameters(),
            *self.a_enc.parameters(),
            *self.dec.parameters(),
            *self.vq.parameters(),
        ], lr=self.learning_rate)
        
        return dict(
            optimizer=opt,
            lr_scheduler=dict(
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, factor=self.learning_rate_schedule_factor,
                ),
                monitor="val/loss",
            ),
        )
    
    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        a,t,p,x = copy.deepcopy(batch)
        
        emb_loss, rec = self.compute_losses(a,t,p,x, timing_dropout=self.timing_dropout)
        loss = emb_loss + rec
        
        self.log("train/emb_loss", emb_loss.detach(), logger=True, on_step=True, on_epoch=False)
        self.log("train/rec", rec.detach(), logger=True, on_step=True, on_epoch=False)
        self.log("train/loss", loss.detach(), logger=True, on_step=True, on_epoch=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        a,t,p,x = copy.deepcopy(batch)

        emb_loss, rec = self.compute_losses(a,t,p,x, timing_dropout=self.timing_dropout, pad=True)
        loss = emb_loss + rec
        
        self.log("val/emb_loss", emb_loss.detach(), logger=True, on_step=False, on_epoch=True, batch_size=1)
        self.log("val/rec", rec.detach(), logger=True, on_step=False, on_epoch=True, batch_size=1)
        self.log("val/loss", loss.detach(), logger=True, on_step=False, on_epoch=True, batch_size=1)
        
        return a,t,p,x
        
    def validation_epoch_end(self, val_outs):
        if not USE_MATPLOTLIB or len(val_outs) == 0:
            return

        torch.cuda.empty_cache()
        a,t,p,x = copy.deepcopy(val_outs[0])

        samples = self(
            x.repeat(2,1,1),
            a.repeat(2,1,1),
            torch.cat([ t,p ], dim=0),
        ).cpu().numpy()

        x: "X,L" = x.squeeze(0).cpu().numpy()
        
        fig, axs = plt.subplots(
            len(samples) + 1,
            figsize=(x.shape[-1]/150, 3),
            sharex=True,
        )
        
        for sample, ax in zip((x, *samples), axs):
            mu = np.mean(sample)
            sig = np.std(sample)

            ax.set_xticks(np.arange(0,a.shape[-1],100), minor=True)
            ax.set_xticks(np.arange(0,a.shape[-1],1000), minor=False)
            ax.set_yticks(np.linspace(-1,1,5))
            ax.grid(which='both')

            ax.set_ylim((mu-3*sig, mu+3*sig))
            ax.set_xlim((0,a.shape[-1]))
            
            for v in sample:
                ax.plot(v)

        fig.tight_layout()
        self.logger.experiment.add_figure("reconstructions", fig, global_step=self.global_step)
        plt.close(fig)