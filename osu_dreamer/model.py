from typing import List, Optional, Tuple

from pathlib import Path
from functools import partial
from multiprocessing import Pool
import random
import copy

from tqdm import tqdm

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange, Reduce
from einops import rearrange

import numpy as np
import librosa

try:
    import matplotlib.pyplot as plt
    USE_MATPLOTLIB = True
except:
    USE_MATPLOTLIB = False

import pytorch_lightning as pl
from torch.utils.data import IterableDataset, DataLoader, random_split

from osu_dreamer.osu.beatmap import Beatmap
from osu_dreamer.signal import (
    MAP_SIGNAL_DIM as X_DIM,
    TIMING_DIM as T_DIM,
    from_beatmap as signal_from_beatmap,
    timing_signal as beatmap_timing_signal,
)

import scipy.stats

# audio processing constants
N_FFT = 2048
HOP_LEN_S = 128. / 22050 # ~6 ms per frame
N_MELS = 64

# model constants
A_DIM = 40
VALID_PAD = 1024


#
#
# =============================================================================
# MODULES
# =============================================================================
#
#

random_hex_string = lambda num: hex(random.randrange(16**num))[2:]

exists = lambda x: x is not None

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim):
    return nn.ConvTranspose1d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv1d(dim, dim, 4, 2, 1, padding_mode='reflect')

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, max_timescale=10000):
        super().__init__()
        assert dim % 2 == 0
        self.fs = torch.pow(max_timescale, torch.linspace(0, -1, dim // 2))

    def forward(self, t: "N,") -> "N,T":
        # {sin,cos}(t / max_timescale^[0..1])
        embs = t[:, None] * self.fs.to(t.device)[None, :]
        embs = torch.cat([embs.sin(), embs.cos()], dim=-1)
        return embs
    
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        h_dim = dim_head * heads
        self.dim_head = dim_head
        
        self.to_qkv: "N,C,L -> N,3H,L" = nn.Conv1d(dim, h_dim*3, 1, bias=False)
        self.to_out = nn.Conv1d(h_dim, dim, 1)

    def forward(self, x: "N,C,L") -> "N,C,L":
        n, c, l = x.shape
        
        qkv: "3,N,h_dim,L" = self.to_qkv(x).chunk(3, dim=1)
        out = self.attn(*( t.unflatten(1, (self.heads, -1)) for t in qkv ))
        return self.to_out(out)
        
    def attn(self, q: "N,h,d,L", k: "N,h,d,L", v: "N,h,d,L"):
        q = q * self.scale

        sim: "N,h,L,L" = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out:"N,h,L,d" = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h l c -> b (h c) l")
        return out
        
    
class WaveBlock(nn.Module):
    """context is acquired from num_stacks*2**stack_depth neighborhood"""
    
    def __init__(self, dim, stack_depth, num_stacks):
        super().__init__()
        
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    dim, 2 * dim,
                    kernel_size=2,
                    padding=2**i,
                    dilation=2**(i+1),
                    padding_mode='reflect',
                ),
                nn.GLU(dim=1),
            )
            for _ in range(num_stacks)
            for i in range(stack_depth)
        ])
        
    def forward(self, x: "N,C,L") -> "N,C,L":
        h = x
        for net in self.nets:
            h = net(h)
            x = x + h
        return x

class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, emb_dim=None, mult=2, norm=True, groups=1):
        super().__init__()
        
        self.mlp = (
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(emb_dim, dim),
            )
            if exists(emb_dim)
            else None
        )

        self.ds_conv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim, padding_mode='reflect')

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv1d(dim, dim_out * mult, 3,1,1, padding_mode='reflect', groups=groups),
            nn.SiLU(),

            nn.GroupNorm(1, dim_out * mult),
            nn.Conv1d(dim_out * mult, dim_out, 3,1,1, padding_mode='reflect', groups=groups),
        )

        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: "N,C,L", time_emb: "N,T" = None) -> "N,D,L":
        h: "N,C,L" = self.ds_conv(x)

        if exists(self.mlp) and exists(time_emb):
            condition: "N,C" = self.mlp(time_emb)
            h = h + condition.unsqueeze(-1)

        h: "N,D,L" = self.net(h)
        return h + self.res_conv(x)
    
class UNet(nn.Module):
    def __init__(
        self,
        h_dim,
        h_dim_groups,
        dim_mults,
        convnext_mult,
        wave_stack_depth,
        wave_num_stacks,
    ):
        super().__init__()
        
        block = partial(ConvNextBlock, mult=convnext_mult, groups=h_dim_groups)
        
        self.init_conv = nn.Sequential(
            nn.Conv1d(X_DIM+A_DIM+T_DIM, h_dim, 7, padding=3),
            WaveBlock(h_dim, wave_stack_depth, wave_num_stacks),
        )

        dims = [h_dim, *(h_dim*m for m in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_layers = len(in_out)
        
        # time embeddings
        emb_dim = h_dim * 4
        self.time_mlp: "N, -> N,T" = nn.Sequential(
            SinusoidalPositionEmbeddings(h_dim),
            nn.Linear(h_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # layers
        
        self.downs = nn.ModuleList([
            nn.ModuleList([
                block(dim_in, dim_out, emb_dim=emb_dim),
                block(dim_out, dim_out, emb_dim=emb_dim),
                Downsample(dim_out) if ind < (num_layers - 1) else nn.Identity(),
            ])
            for ind, (dim_in, dim_out) in enumerate(in_out)
        ])

        mid_dim = dims[-1]
        self.mid_block1 = block(mid_dim, mid_dim, emb_dim=emb_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block(mid_dim, mid_dim, emb_dim=emb_dim)
        
        self.ups = nn.ModuleList([
            nn.ModuleList([
                block(dim_out * 2, dim_in, emb_dim=emb_dim),
                block(dim_in, dim_in, emb_dim=emb_dim),
                Upsample(dim_in) if ind < (num_layers - 1) else nn.Identity(),
            ])
            for ind, (dim_in, dim_out) in enumerate(in_out[::-1])
        ])

        self.final_conv = nn.Sequential(
            block(h_dim, h_dim),
            zero_module(nn.Conv1d(h_dim, X_DIM, 1)),
        )
        

    def forward(self, x: "N,X,L", a: "N,A,L", t: "N,T,L", ts: "N,") -> "N,X,L":
        
        x: "N,X+A,L" = torch.cat((x,a,t), dim=1)
        
        x: "N,h_dim,L" = self.init_conv(x)

        h = []
        emb: "N,T" = self.time_mlp(ts)

        # downsample
        for block1, block2, downsample in self.downs:
            x: "N,out,L" = block1(x, emb)
            x: "N,out,L" = block2(x, emb)
            h.append(x)
            x: "N,out,L//2" = downsample(x)

        # bottleneck
        x: "N,mid,L" = self.mid_block1(x, emb)
        x: "N,mid,L" = self.mid_attn(x)
        x: "N,mid,L" = self.mid_block2(x, emb)

        # upsample
        for block1, block2, upsample in self.ups:
            x: "N,2*out,L" = torch.cat((x, h.pop()), dim=1)
            x: "N,in,L" = block1(x, emb)
            x: "N,in,L" = block2(x, emb)
            x: "N,in,L*2" = upsample(x)

        return self.final_conv(x)
    
#
#
# =============================================================================
# MODEL DEFINITION
# =============================================================================
#
#


def extract(a, ts, x_shape):
    batch_size = ts.shape[0]
    out = a.gather(-1, ts.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(ts.device)
    
class BetaSchedule:
    def __init__(self, betas, net):
        self._net = net
        
        # define beta schedule
        self.betas = betas
        self.timesteps = len(betas)

        # define alphas 
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.rsqrt(self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        
        # Improved DDPM, Eqn. 10
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod) # beta_tilde
        assert (self.posterior_variance[1:] != 0).all(), self.posterior_variance[1:]
        
    def net(self, x,a,t,ts):
        return self._net(x,a,t,ts)

    def q_sample(self, x: "N,X,L", ts: "N,", noise=None) -> "N,X,L":
        """sample q(x_t|x_0) using the nice property"""
        if noise is None:
            noise = torch.randn_like(x)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, ts, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, ts, x.shape
        )

        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
    
    def model_eps_var(self, x, a, t, ts):
        model_eps = self.net(x,a,t,ts)
        model_var = extract(self.posterior_variance, ts, x.shape)
        return model_eps, model_var
    
    def p_eps_mean_var(self, x, a, t, ts):
        """sample from p(x_{t-1} | x_t)"""
        model_eps, model_var = self.model_eps_var(x,a,t,ts)
            
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        betas_t = extract(self.betas, ts, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, ts, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, ts, x.shape)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model_eps / sqrt_one_minus_alphas_cumprod_t)
        
        return model_eps, model_mean, model_var
        
    @torch.no_grad()
    def sample(self, a: "N,A,L", t: "N,T,L", x: "N,X,L" = None, *, ddim=False) -> "N,X,L":
        """sample p(x)"""
        
        b,_,l = a.size()
        
        if x is None:
            # start from pure noise (for each example in the batch)
            x = torch.randn((b,X_DIM,l), device=a.device)

        print()
        for i in tqdm(list(reversed(range(self.timesteps))), desc='sampling loop time step'):
            ts = torch.full((b,), i, device=a.device, dtype=torch.long)
            
            _, model_mean, model_var = self.p_eps_mean_var(x,a,t,ts)

            if i == 0 or ddim:
                x = model_mean
            else:
                x = model_mean + torch.sqrt(model_var) * torch.randn_like(x)
            
        return x
    
class CosineBetaSchedule(BetaSchedule):
    def __init__(self, timesteps, net, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)

        super().__init__(betas, net)

    
class StridedBetaSchedule(BetaSchedule):
    def __init__(self, schedule, steps, *args, **kwargs):
        # use_timesteps = set(torch.linspace(1, schedule.timesteps, steps).round().int().tolist())
        use_timesteps = set(torch.arange(1,schedule.timesteps, schedule.timesteps/steps).round().int().tolist())
        self.ts_map = []

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(schedule.alphas_cumprod):
            if i in use_timesteps:
                self.ts_map.append(i)
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                
        super().__init__(torch.tensor(new_betas), *args, **kwargs)
                
            
    def net(self, x,a,t,ts):
        ts = torch.tensor(self.ts_map, device=ts.device, dtype=ts.dtype)[ts]
        return super().net(x,a,t,ts)
    

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
        import shutil
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
    
    def training_step(self, batch: ("N,A,L", "N,T,L", "N,T,L", "N,X,L"), batch_idx):
        torch.cuda.empty_cache()
        a,t,p,x = copy.deepcopy(batch)
        
        loss = self.compute_loss(a,t,p,x,timing_dropout=self.timing_dropout)
        
        self.log(
            "train/loss", loss.detach(),
            logger=True, on_step=True, on_epoch=False,
        )
        
        return loss

    def validation_step(self, batch: ("1,A,L","1,T,L","1,T,L","1,X,L"), batch_idx, *args, **kwargs):
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
        

#
#
# =============================================================================
# DATA
# =============================================================================
#
#

import os

# check if using WSL
if os.system("uname -r | grep microsoft > /dev/null") == 0:
    def reclaim_memory():
        """
        free the vm page cache - see `https://devblogs.microsoft.com/commandline/memory-reclaim-in-the-windows-subsystem-for-linux-2/`
        
        add to /etc/sudoers:
        %sudo ALL=(ALL) NOPASSWD: /bin/tee /proc/sys/vm/drop_caches
        """
        os.system("echo 1 | sudo tee /proc/sys/vm/drop_caches > /dev/null")
else:
    def reclaim_memory():
        pass

def load_audio(audio_file):
    wave, sr = torchaudio.load(audio_file, normalize=True)
    if wave.dim() > 1:
        wave = wave.mean((0,))

    # compute spectrogram
    spec: "A,L" = torchaudio.transforms.MFCC(
        sample_rate=sr, 
        melkwargs=dict(
            normalized=True,
            n_fft=N_FFT,
            hop_length=int(HOP_LEN_S * sr),
            n_mels=N_MELS,
        ),
    )(wave).numpy()
    
    return spec, sr

def prepare_map(data_dir, map_file):
    try:
        bm = Beatmap(map_file, meta_only=True)
    except Exception as e:
        print(f"{map_file}: {e}")
        return

    af_dir = "_".join([bm.audio_filename.stem, *(s[1:] for s in bm.audio_filename.suffixes)])
    map_dir = data_dir / map_file.parent.name / af_dir
    
    spec_path =  map_dir / "spec.pt"
    timing_path = map_dir / "timing.pt"
    plp_path = map_dir / "plp.pt"
    map_path = map_dir / f"{map_file.stem}.map.pt"
    
    if map_path.exists():
        return
    
    try:
        bm.parse_map_data()
    except Exception as e:
        print(f"{map_file}: {e}")
        return

    if spec_path.exists():
        # determine audio sample rate
        sr = torchaudio.info(bm.audio_filename).sample_rate
        spec = np.load(spec_path)
    else:
        # load audio file
        try:
            spec, sr = load_audio(bm.audio_filename)
        except Exception as e:
            print(f"{bm.audio_filename}: {e}")
            return

        # save spectrogram
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        with open(spec_path, "wb") as f:
            np.save(f, spec, allow_pickle=False)
            
    frame_times = librosa.frames_to_time(
        np.arange(spec.shape[-1]),
        sr=sr, hop_length=int(HOP_LEN_S * sr), n_fft=N_FFT,
    ) * 1000
    
    # compute timing signal
    if not timing_path.exists():
        timing: "T,L" = beatmap_timing_signal(bm, frame_times)
        with open(timing_path, "wb") as f:
            np.save(f, timing, allow_pickle=False)
            
    # compute plp
    if not plp_path.exists():
        plp: "T,L" = librosa.beat.plp(
            onset_envelope=librosa.onset.onset_strength(
                S=spec, center=False,
            ),
            prior = scipy.stats.lognorm(loc=np.log(180), scale=180, s=1),
            # use 10s of audio to determine local bpm
            win_length=int(10. / HOP_LEN_S), 
        )[None]
        with open(plp_path, "wb") as f:
            np.save(f, plp, allow_pickle=False)

    # compute map signal
    x: "X,L" = signal_from_beatmap(bm, frame_times)
    with open(map_path, "wb") as f:
        np.save(f, x, allow_pickle=False)
    

class Data(pl.LightningDataModule):
    def __init__(
        self,
        
        seq_depth: int,
        sample_density: float,
        subseq_density: float,
        batch_size: int,
        num_workers: int,
        
        data_path: str = "./data",
        src_path: str = None,
        val_split: float = None,
        val_size: int = None,
    ):
        super().__init__()
        
        self.seq_len = 2 ** seq_depth
        
        self.sample_density = sample_density
        self.subseq_density = subseq_density
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        if (val_split is None) == (val_size is None):
            raise ValueError('exactly one of `val_split` or `val_size` must be specified')
        self.val_split = val_split
        self.val_size = val_size
        
        self.data_dir = Path(data_path)
        try:
            self.data_dir.mkdir()
        except FileExistsError:
            # data dir exists, check for samples
            try:
                next(self.data_dir.rglob("*.map.pt"))
                # data dir already has samples
                self.src_dir = None
                return
            except StopIteration:
                # data dir is empty, must prepare data
                pass
            
        if src_path is None:
            raise ValueError("`src_path` must be specified when `data_dir` does not exist or is empty")
        self.src_dir = Path(src_path)
        
        
    def prepare_data(self):
        if self.src_dir is None:
            return
        
        src_maps = list(self.src_dir.rglob("*.osu"))
        print(f"{len(src_maps)} osu! beatmaps found, processing...")
        with Pool(processes=self.num_workers) as p:
            for _ in tqdm(p.imap_unordered(partial(prepare_map, self.data_dir), src_maps), total=len(src_maps)):
                reclaim_memory()
            
    def setup(self, stage: str):
        full_set = list(self.data_dir.rglob("*.map.pt"))
        
        if self.val_size is not None:
            val_size = self.val_size
        else:
            val_size = int(len(full_set) * self.val_split)
            
        if val_size < 0:
            raise ValueError("`val_size` is negative")
            
        if val_size > len(full_set):
            raise ValueError(f"`val_size` ({val_size}) is greater than the number of samples ({len(full_set)})")
            
        train_size = len(full_set) - val_size
        print(f'train: {train_size} | val: {val_size}')
        train_split, val_split = random_split(full_set, [train_size, val_size])
        
        self.train_set = SubsequenceDataset(
            dataset=train_split,
            seq_len=self.seq_len,
            sample_density=self.sample_density,
            subseq_density=self.subseq_density,
        )
        self.val_set = FullSequenceDataset(dataset=val_split)
            
    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=1,
            num_workers=self.num_workers,
        )
    
    
class StreamPerSample(IterableDataset):
    def __init__(self, **kwargs):
        super().__init__()
        self.dataset = kwargs.pop("dataset")
        self.sample_density = kwargs.pop("sample_density", 1.)
        
        if not 0 < self.sample_density <= 1:
            raise ValueError("sample density must be in (0, 1]:", self.sample_density)
            
        if len(kwargs):
            raise ValueError(f"unexpected kwargs: {kwargs}")
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  
            # single-process data loading, return the full iterator
            num_workers = 1
            worker_id = 0
            seed = torch.initial_seed()
        else:  # in a worker process
            # split workload
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            seed = worker_info.seed
        
        random.seed(seed)
        
        dataset = sorted(self.dataset)
        for i, sample in random.sample(list(enumerate(dataset)), int(len(dataset) * self.sample_density)):
            if i % num_workers != worker_id:
                continue
                
            try:
                for x in self.sample_stream(sample):
                    yield x
            finally:
                reclaim_memory()
                
                
def load_tensors_for_map(map_file):
    a: "A,L" = torch.tensor(np.load(map_file.parent / "spec.pt")).float()
    t: "T,L" = torch.tensor(np.load(map_file.parent / "timing.pt")).float()
    p: "T,L" = torch.tensor(np.load(map_file.parent / "plp.pt")).float()
    x: "X,L" = torch.tensor(np.load(map_file)).float()
    return a,t,p,x

class FullSequenceDataset(StreamPerSample):
    MAX_LEN = 60000
            
    def sample_stream(self, map_file):
        yield tuple([ 
            x[...,:self.MAX_LEN]
            # x[...,:self.MAX_LEN] if x.size(-1) > self.MAX_LEN else F.pad(x, (0, self.MAX_LEN - x.size(-1)))
            for x in load_tensors_for_map(map_file)
        ])     
        
class SubsequenceDataset(StreamPerSample):
    def __init__(self, **kwargs):
        self.seq_len = kwargs.pop("seq_len")
        self.subseq_density = kwargs.pop("subseq_density", 2)
        super().__init__(**kwargs)

    def sample_stream(self, map_file):
        tensors = load_tensors_for_map(map_file)
        L = tensors[0].size(-1)

        if self.seq_len >= L:
            return

        num_samples = int(L / self.seq_len * self.subseq_density)

        for idx in torch.randperm(L - self.seq_len)[:num_samples]:
            yield tuple([
                x[..., idx:idx+self.seq_len].clone()
                for x in tensors
            ])