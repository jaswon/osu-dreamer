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
from osu_dreamer.signal import MAP_SIGNAL_DIM as X_DIM

from .vq import VectorQuantizer
from ..modules import ConvNextBlock, WaveBlock, Downsample, Upsample, Residual, PreNorm, Attention
    
class Encoder(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        h_dims: List[int],
        groups: int, 
        convnext_mult: int,
        wave_stack_depth: int,
        wave_num_stacks: int,
    ):
        super().__init__(
            nn.Conv1d(input_dim, h_dims[0], 7,1,3),
            *(
                layer for dim_in, dim_out in zip(h_dims[:-1], h_dims[1:])
                for layer in [
                    WaveBlock(dim_in, wave_stack_depth, wave_num_stacks, mult=convnext_mult, h_dim_groups=groups),
                    ConvNextBlock(dim_in, dim_out, mult=convnext_mult, groups=groups),
                    Downsample(dim_out),
                ]
            ),
        )


class UNetDecoder(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        h_dims,
        groups,
        convnext_mult,
        wave_stack_depth,
        wave_num_stacks,
    ):
        super().__init__()
        
        # layers

        self.init_conv = nn.Sequential(
            nn.Conv1d(input_dim, h_dims[0], 7, padding=3),
        )

        self.downs = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    WaveBlock(dim_in, wave_stack_depth, wave_num_stacks, mult=convnext_mult, h_dim_groups=groups),
                    ConvNextBlock(dim_in, dim_out, mult=convnext_mult, groups=groups),
                ),
                Downsample(dim_out),
            ])
            for dim_in, dim_out in zip(h_dims[:-1], h_dims[1:])
        ])

        mid_dim = h_dims[-1] * 2
        self.mid_net = nn.Sequential(
            ConvNextBlock(mid_dim, mid_dim, mult=convnext_mult, groups=groups),
            ConvNextBlock(mid_dim, h_dims[-1], mult=convnext_mult, groups=groups),
        )
        
        self.ups = nn.ModuleList([
            nn.ModuleList([
                Upsample(dim_in),
                nn.Sequential(
                    ConvNextBlock(dim_in * 2, dim_out, mult=convnext_mult, groups=groups),
                    ConvNextBlock(dim_out, dim_out, mult=convnext_mult, groups=groups),
                ),
            ])
            for dim_in, dim_out in zip(h_dims[-1:0:-1], h_dims[-2::-1])
        ])

        self.final_conv = nn.Sequential(
            nn.Conv1d(h_dims[0], h_dims[0], 3,1,1),
            nn.Conv1d(h_dims[0], output_dim, 3,1,1),
            nn.Conv1d(output_dim, output_dim, 3,1,1),
        )
        

    def forward(self, a: "N,A,L", x: "N,H,l") -> "N,X,L":
        a = self.init_conv(a)
        h = []
        # downsample
        for net, downsample in self.downs:
            a = net(a)
            h.append(a)
            a = downsample(a)

        x = torch.cat([a,x], dim=1)

        # bottleneck
        x = self.mid_net(x)

        # upsample
        for upsample, net in self.ups:
            x = net(torch.cat([upsample(x), h.pop()], dim=1))

        return self.final_conv(x)
    
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
        h_dim_groups: int,
        convnext_mult: int,
        wave_stack_depth: int,
        wave_num_stacks: int,
        n_embed: int,
        
        learning_rate: float = 0.,
        learning_rate_schedule_factor: float = 0.,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.learning_rate = learning_rate
        self.learning_rate_schedule_factor = learning_rate_schedule_factor

        self.depth = len(h_dims)

        # models
        
        self.enc = Encoder(X_DIM+A_DIM, h_dims, h_dim_groups, convnext_mult, wave_stack_depth, wave_num_stacks)
        self.dec = UNetDecoder(A_DIM, X_DIM, h_dims, h_dim_groups, convnext_mult, wave_stack_depth, wave_num_stacks)
        self.vq  = VectorQuantizer(n_embed, h_dims[-1])
        
    def inference_pad(self, x):
        x = F.pad(x, (VALID_PAD, VALID_PAD), mode='replicate')
        pad = (1 + x.size(-1) // 2 ** self.depth) * 2 ** self.depth - x.size(-1)
        x = F.pad(x, (0, pad), mode='replicate')
        return x, (..., slice(VALID_PAD,-(VALID_PAD+pad)))
    
    def decode(self, a: "N,A,L", z: "N,H,L") -> "N,X,L":
        z_q, _, _ = self.vq(z)
        x_hat = self.dec(a, z_q)
        x_hat[:, :-2] = torch.sigmoid(x_hat[:, :-2])*2-1
        return x_hat

    def forward(self, a: "N,A,L", x: "N,X,L"):
        """x_hat"""
        a, sl = self.inference_pad(a)
        x, _ = self.inference_pad(x)

        z = self.enc(torch.cat([a,x], dim=1))
        x_hat = self.decode(a, z)[sl]
        return x_hat
    
    def compute_losses(self, a: "N,A,L", x: "N,X,L", pad=False):
        if pad:
            a, _ = self.inference_pad(a)
            x, _ = self.inference_pad(x)

        # embedding loss
        z = self.enc(torch.cat([a,x], dim=1))
        z_q, emb_loss, _ = self.vq(z)
        
        # reconstruction loss
        x_hat = self.dec(a, z_q)
        rec_loss = F.binary_cross_entropy_with_logits(x_hat[:,:-2], (x[:,:-2]+1)/2) + F.mse_loss(x_hat[:,-2:], x[:,-2:])

        return emb_loss, rec_loss
     
#
#
# ====================================
# MODEL TRAINING
# ====================================
#
#

    def configure_optimizers(self):
        opt = torch.optim.AdamW([
            *self.enc.parameters(),
            *self.dec.parameters(),
            *self.vq.parameters(),
        ], lr=self.learning_rate)
        
        return dict(
            optimizer=opt,
            lr_scheduler=dict(
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, factor=self.learning_rate_schedule_factor,
                ),
                monitor="val/rec",
            ),
        )
    
    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        a,x = copy.deepcopy(batch)
        
        emb_loss, rec = self.compute_losses(a,x)
        loss = emb_loss + rec
        
        self.log("train/emb_loss", emb_loss.detach(), logger=True, on_step=True, on_epoch=False)
        self.log("train/rec", rec.detach(), logger=True, on_step=True, on_epoch=False)
        self.log("train/loss", loss.detach(), logger=True, on_step=True, on_epoch=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        a,x = copy.deepcopy(batch)

        emb_loss, rec = self.compute_losses(a,x,pad=True)
        loss = emb_loss + rec
        
        self.log("val/emb_loss", emb_loss.detach(), logger=True, on_step=False, on_epoch=True, batch_size=1)
        self.log("val/rec", rec.detach(), logger=True, on_step=False, on_epoch=True, batch_size=1)
        self.log("val/loss", loss.detach(), logger=True, on_step=False, on_epoch=True, batch_size=1)
        
        return a,x
        
    def validation_epoch_end(self, val_outs):
        if not USE_MATPLOTLIB or len(val_outs) == 0:
            return

        torch.cuda.empty_cache()
        a,x = copy.deepcopy(val_outs[0])

        x_hat = self(a,x).squeeze(0).cpu().numpy()
        x: "X,L" = x.squeeze(0).cpu().numpy()
        
        fig, axs = plt.subplots(
            nrows=2,
            figsize=(x.shape[-1]/300, 2),
            sharex=True,
        )
        
        for sample, ax in zip((x, x_hat), axs):
            ax.set_ylim((-1.5,1.5))
            ax.set_xlim((0,a.shape[-1]))
            
            for v in sample:
                ax.plot(v)

        fig.tight_layout()
        self.logger.experiment.add_figure("reconstructions", fig, global_step=self.global_step)
        plt.close(fig)