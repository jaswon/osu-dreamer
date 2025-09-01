from typing import Any
from jaxtyping import Float, Int

import torch as th
from torch import Tensor, nn

import pytorch_lightning as pl

from osu_dreamer.modules.muon import Muon
from osu_dreamer.modules.lr_schedule import LRScheduleArgs, make_lr_schedule
from osu_dreamer.data.load_audio import get_frame_times

from .data.dataset import Batch
from .data.tokens.tokens import VocabConfig, make_vocab

from osu_dreamer.modules.spec_features import SpecFeatures
from .modules.multiscale_ctx import MultiScaleEncoder
from .modules.global_ctx import GlobalEncoder
from .modules.decoder import Decoder, DecoderArgs


class Model(pl.LightningModule):
    def __init__(
        self,

        # training parameters
        opt_args: dict[str, Any],
        schedule_args: LRScheduleArgs,
        
        # model hparams
        vocab_config: VocabConfig,
        emb_dim: int,
        decoder_args: DecoderArgs,
        
        # audio encoder hparams
        ctx_dim: int,
        audio_h_dim: int,
        num_global_ctx: int,
        ctx_scales: list[tuple[int, int]],
        
        # validation parameters
        val_batches: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # training params
        self.opt_args = opt_args
        self.lr_schedule = make_lr_schedule(schedule_args)
        
        # validation params
        self.val_batches = val_batches
        
        # model components
        self.vocab = make_vocab(vocab_config)
        vocab_size = len(self.vocab)
        self.token_embed = nn.Embedding(vocab_size, emb_dim)
        self.decoder = Decoder(emb_dim, ctx_dim, decoder_args)
        self.token_head = nn.Linear(emb_dim, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 = PAD token
        
        # audio encoder
        self.audio_encoder = SpecFeatures(audio_h_dim)
        self.global_encoder = GlobalEncoder(audio_h_dim, ctx_dim, num_global_ctx)
        self.ctx_encoder = MultiScaleEncoder(audio_h_dim, ctx_dim, ctx_scales)
        
    
    def forward(
        self,
        audio: Float[Tensor, "A L"],
        map_features: Float[Tensor, "M"],
        tokens: Int[Tensor, "B N+1"],
        timestamps: Int[Tensor, "B N"],
    ) -> tuple[
        Float[Tensor, "B N V"], # pred logits
        Int[Tensor, "B N"]      # true targets
    ]:
    
        audio_features = self.audio_encoder(audio)  # 1 D L
        
        global_ctx = self.global_encoder(audio_features) # G C
        frame_times = th.tensor(get_frame_times(audio_features.size(-1)), device=audio.device)  # L
        frame_idxs = th.searchsorted(frame_times, timestamps) # B N
        multi_scale_ctx = self.ctx_encoder(self.ctx_encoder.precompute(audio_features), frame_idxs) # B N T C

        expanded_global_ctx = global_ctx[None, None, ...].expand(tokens.size(0), timestamps.size(1), -1, -1)
        ctx = th.cat([ expanded_global_ctx, multi_scale_ctx ], dim=2) # B N T+G C
        
        embs = self.token_embed(tokens[:,:-1]) # B N D

        output, _ = self.decoder(embs, ctx=ctx)
        logits = self.token_head(output) # B N V
        
        return logits, tokens[:,1:]
    
    def training_step(self, batch: Batch, batch_idx: int):
        # Forward pass
        pred_logits, target_tokens = self.forward(
            batch.audio,
            batch.map_features,
            batch.tokens,
            batch.timestamps,
        )
        
        loss = self.criterion(pred_logits.reshape(-1, pred_logits.size(-1)), target_tokens.reshape(-1))
        
        # Log metrics
        self.log('train/loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch: Batch, batch_idx: int):
        if batch_idx >= self.val_batches:
            return
        
        # Forward pass
        pred_logits, target_tokens = self.forward(
            batch.audio,
            batch.map_features,
            batch.tokens,
            batch.timestamps,
        )
        
        # Calculate loss
        loss = self.criterion(pred_logits.reshape(-1, pred_logits.size(-1)), target_tokens.reshape(-1))
        
        # Calculate accuracy
        pred_tokens = pred_logits.argmax(dim=-1)
        accuracy = (pred_tokens == target_tokens).float().mean()
        
        # Log metrics
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/accuracy', accuracy, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = Muon(self.parameters(), **self.opt_args)
        
        scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, self.lr_schedule)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        } 