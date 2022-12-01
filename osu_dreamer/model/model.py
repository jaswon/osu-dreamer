from typing import List, Tuple

from pathlib import Path
import random
import copy

import numpy as np
import scipy.stats
import librosa

import torch
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    USE_MATPLOTLIB = True
except:
    USE_MATPLOTLIB = False

import pytorch_lightning as pl

from osu_dreamer.signal import (
    MAP_SIGNAL_DIM as X_DIM,
    TIMING_DIM as T_DIM,
    timing_signal as beatmap_timing_signal,
)

from .data import N_FFT, HOP_LEN_S, load_audio
from .beta_schedule import CosineBetaSchedule, StridedBetaSchedule
from .modules import UNet

# model constants
A_DIM = 40
VALID_PAD = 1024
    

class Model(pl.LightningModule):
    def __init__(
        self,
        h_dim: int,
        h_dim_groups: int,
        dim_mults: List[int],
        convnext_mult: int,
        wave_stack_depth: int,
        wave_num_stacks: int,
        
        timesteps: int,
        sample_steps: int,
    
        loss_type: str,
        timing_dropout: float,
        learning_rate: float = 0.,
        learning_rate_schedule_factor: float = 0.,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # model
        self.net = UNet(
            h_dim, h_dim_groups, dim_mults, 
            convnext_mult,
            wave_stack_depth,
            wave_num_stacks,
        )
        
        self.schedule = CosineBetaSchedule(timesteps, self.net)
        self.sampling_schedule = StridedBetaSchedule(self.schedule, sample_steps, self.net)
        
        # training params
        try:
            self.loss_fn = dict(
                l1 = F.l1_loss,
                l2 = F.mse_loss,
                huber = F.smooth_l1_loss,
            )[loss_type]
        except KeyError:
            raise NotImplementedError(loss_type)

        self.learning_rate = learning_rate
        self.learning_rate_schedule_factor = learning_rate_schedule_factor
        self.timing_dropout = timing_dropout
        self.depth = len(dim_mults)
    
#
#
# =============================================================================
# MODEL INFERENCE
# =============================================================================
#
#
        
    def inference_pad(self, x):
        x = F.pad(x, (VALID_PAD, VALID_PAD), mode='replicate')
        pad = (1 + x.size(-1) // 2 ** self.depth) * 2 ** self.depth - x.size(-1)
        x = F.pad(x, (0, pad), mode='replicate')
        return x, (..., slice(VALID_PAD,-(VALID_PAD+pad)))
        
    def forward(self, a: "N,A,L", t: "N,T,L", **kwargs):
        a, sl = self.inference_pad(a)
        t, _  = self.inference_pad(t)
        return self.sampling_schedule.sample(a, t, **kwargs)[sl]
    
    def generate_mapset(
        self,
        audio_file,
        timing,
        num_samples,
        title,
        artist,
    ):
        from zipfile import ZipFile
        from osu_dreamer.signal import to_beatmap as signal_to_map
        
        metadata = dict(
            audio_filename=audio_file.name,
            title=title,
            artist=artist,
        )
        
        # load audio
        # ======
        dev = next(self.parameters()).device
        a, sr = load_audio(audio_file)
        a = torch.tensor(a, device=dev)

        frame_times = librosa.frames_to_time(
            np.arange(a.shape[-1]),
            sr=sr, hop_length=int(HOP_LEN_S * sr), n_fft=N_FFT,
        ) * 1000
        
        # generate maps
        # ======
        
        # `timing` can be one of:
        # - List[TimingPoint] : timed according to timing points
        # - number : audio is constant known BPM
        # - None : no prior knowledge of audio timing
        if isinstance(timing, list):
            t = torch.tensor(beatmap_timing_signal(timing, frame_times), device=dev).float()
        else:
            if timing is None:
                bpm_prior = scipy.stats.lognorm(loc=np.log(180), scale=180, s=1)
            else:
                bpm_prior = scipy.stats.norm(loc=timing, scale=1)
                
            t = torch.tensor(librosa.beat.plp(
                onset_envelope=librosa.onset.onset_strength(
                    S=a.cpu().numpy(), center=False,
                ),
                prior = bpm_prior,
                # use 10s of audio to determine local bpm
                win_length=int(10. / HOP_LEN_S), 
            )[None], device=dev)
            
        
        pred_signals = self(
            a.repeat(num_samples,1,1),
            t[None, :].repeat(num_samples,1,1),
        ).cpu().numpy()

        random_hex_string = lambda num: hex(random.randrange(16**num))[2:]
        
        # package mapset
        # ======
        while True:
            mapset = Path(f"_{random_hex_string(7)} {artist} - {title}.osz")
            if not mapset.exists():
                break
                
        with ZipFile(mapset, 'x') as mapset_archive:
            mapset_archive.write(audio_file, audio_file.name)
            
            for i, pred_signal in enumerate(pred_signals):
                mapset_archive.writestr(
                    f"{artist} - {title} (osu!dreamer) [version {i}].osu",
                    signal_to_map(
                        dict( **metadata, version=f"version {i}" ),
                        pred_signal, frame_times, copy.deepcopy(timing),
                    ),
                )
                    
        return mapset
    
#
#
# =============================================================================
# MODEL TRAINING
# =============================================================================
#
#

    def compute_loss(self, a, t, p, x, pad=False, timing_dropout=0.):
        ts = torch.randint(0, self.schedule.timesteps, (x.size(0),), device=x.device).long()
        
        if pad:
            a, _ = self.inference_pad(a)
            t, _ = self.inference_pad(t)
            p, _ = self.inference_pad(p)
            x, _ = self.inference_pad(x)
            
        if timing_dropout > 0:
            drop_idxs = torch.randperm(t.size(0))[:int(t.size(0) * timing_dropout)]
            t[drop_idxs] = p[drop_idxs]
        
        true_eps: "N,X,L" = torch.randn_like(x)

        x_t: "N,X,L" = self.schedule.q_sample(x, ts, true_eps)
        
        pred_eps = self.net(x_t, a, t, ts)
        
        return self.loss_fn(true_eps, pred_eps).mean()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
        
        return dict(
            optimizer=opt,
            lr_scheduler=dict(
                scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(
                    opt, factor=self.learning_rate_schedule_factor),
                monitor="val/loss",
            ),
        )
    
    def training_step(self, batch: Tuple["N,A,L", "N,T,L", "N,T,L", "N,X,L"], batch_idx):
        torch.cuda.empty_cache()
        a,t,p,x = copy.deepcopy(batch)
        
        loss = self.compute_loss(a,t,p,x,timing_dropout=self.timing_dropout)
        
        self.log(
            "train/loss", loss.detach(),
            logger=True, on_step=True, on_epoch=False,
        )
        
        return loss

    def validation_step(self, batch: Tuple["1,A,L","1,T,L","1,T,L","1,X,L"], batch_idx, *args, **kwargs):
        torch.cuda.empty_cache()
        a,t,p,x = copy.deepcopy(batch)
        
        loss = self.compute_loss(a,t,p,x, pad=True, timing_dropout=self.timing_dropout)
        dropout_loss = self.compute_loss(a,t,p,x, pad=True, timing_dropout=1.)
        
        self.log(
            "val/loss", loss.detach(),
            logger=True, on_step=False, on_epoch=True,
        )
        
        self.log(
            "val/dropout_loss", dropout_loss.detach(),
            logger=True, on_step=False, on_epoch=True,
        )
        
        return a,t,p,x
        
    def validation_epoch_end(self, val_outs: "List[(1,A,L),(1,T,L),(1,T,L),(1,X,L)]"):
        if not USE_MATPLOTLIB or len(val_outs) == 0:
            return
        
        torch.cuda.empty_cache()
        a,t,p,x = copy.deepcopy(val_outs[0])
        
        samples = self(a.repeat(2,1,1), torch.cat([ t,p ], dim=0) ).cpu().numpy()
        
        a: "A,L" = a.squeeze(0).cpu().numpy()
        x: "X,L" = x.squeeze(0).cpu().numpy()
        
        height_ratios = [1.5] + [1] * (1+len(samples))
        w, h = a.shape[-1]/150, sum(height_ratios)/2
        margin, margin_left = .1, .5
        
        fig, (ax1, *axs) = plt.subplots(
            len(height_ratios), 1,
            figsize=(w, h),
            sharex=True,
            gridspec_kw=dict(
                height_ratios=height_ratios,
                hspace=.1,
                left=margin_left/w,
                right=1-margin/w,
                top=1-margin/h,
                bottom=margin/h,
            )
        )
        
        ax1.imshow(librosa.power_to_db(a), origin="lower", aspect='auto')
        
        for sample, ax in zip((x, *samples), axs):
            mu = np.mean(sample)
            sig = np.std(sample)

            ax.set_ylim((mu-3*sig, mu+3*sig))
            
            for v in sample:
                ax.plot(v)

        self.logger.experiment.add_figure("samples", fig, global_step=self.global_step)
        plt.close(fig)
