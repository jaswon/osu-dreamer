from typing import Any
from jaxtyping import Float, Int, Bool

import torch as th
import torch.nn.functional as F
from torch import Tensor, nn

import pytorch_lightning as pl
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import itertools

from osu_dreamer.lm.data.tokens.state import LogitProcessor
from osu_dreamer.modules.muon import Muon
from osu_dreamer.modules.lr_schedule import LRScheduleArgs, make_lr_schedule
from osu_dreamer.data.load_audio import get_frame_times

from .data.dataset import Batch
from .data.tokens.tokens import Vocab, Token, TokenType

from .modules.audio_encoder import SimpleAudioEncoder
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
        vocab: Vocab,
        emb_dim: int,
        decoder_args: DecoderArgs,
        
        # audio encoder hparams
        ctx_dim: int,
        audio_h_dim: int,
        num_global_ctx: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # training params
        self.opt_args = opt_args
        self.lr_schedule = make_lr_schedule(schedule_args)
        
        # model components
        self.vocab = vocab
        vocab_size = len(vocab.tokens)
        self.token_embed = nn.Embedding(vocab_size, emb_dim)
        self.decoder = Decoder(emb_dim, ctx_dim, decoder_args)
        self.token_head = nn.Linear(emb_dim, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 = PAD token
        
        # audio encoder
        self.audio_encoder = SimpleAudioEncoder(audio_h_dim)
        self.global_encoder = GlobalEncoder(audio_h_dim, ctx_dim, num_global_ctx)
        self.ctx_encoder = MultiScaleEncoder(audio_h_dim, ctx_dim, list(vocab.context_radii))
        
    
    def forward(
        self,
        audio: Float[Tensor, "A L"],
        map_features: Float[Tensor, "M"],
        tokens: Int[Tensor, "B Np1"],
        timestamps: Int[Tensor, "B N"],
        valid: Bool[Tensor, "B N V"],
    ) -> tuple[
        Float[Tensor, "B N V"], # pred logits
        Int[Tensor, "B N"]      # true targets
    ]:
    
        audio_features = self.audio_encoder(audio[None])  # 1 D L
        
        global_ctx = self.global_encoder(audio_features) # G C
        frame_times = th.tensor(get_frame_times(audio_features.size(-1)), device=audio.device)  # L
        frame_idxs = th.searchsorted(frame_times, timestamps) # B N
        multi_scale_ctx = self.ctx_encoder(self.ctx_encoder.precompute(audio_features), frame_idxs) # B N T C

        expanded_global_ctx = global_ctx[None, None, ...].expand(tokens.size(0), timestamps.size(1), -1, -1)
        ctx = th.cat([ expanded_global_ctx, multi_scale_ctx ], dim=2) # B N T+G C
        
        embs = self.token_embed(tokens[:,:-1]) # B N D

        output, _ = self.decoder(embs, ctx=ctx)
        logits = self.token_head(output) # B N V
        logits.masked_fill_(~valid, -th.inf)
        
        return logits, tokens[:,1:]
    
    def training_step(self, batch: Batch, batch_idx: int):
        # Forward pass
        pred_logits, target_tokens = self.forward(
            batch.audio,
            batch.map_features,
            batch.tokens,
            batch.timestamps,
            batch.valid,
        )
        
        loss = self.criterion(pred_logits.reshape(-1, pred_logits.size(-1)), target_tokens.reshape(-1))
        
        # Log metrics
        self.log('train/loss', loss, batch_size=batch.tokens.size(0))
        
        return loss
    
    def validation_step(self, batch: Batch, batch_idx: int):
        
        # On the first validation batch of every epoch, generate a sample
        if batch_idx == 0 and self.global_rank == 0:
            generated_token_ids = self.sample(batch.audio, batch.map_features, max_len=512, top_p=0.)
            generated_tokens = [ self.vocab.tokens[int(i.item())] for i in generated_token_ids[0] ]

            exp: SummaryWriter = self.logger.experiment # type: ignore
            sample_text = '\n'.join([ str(event) for event in generated_tokens ])
            exp.add_text(f'sample', sample_text, global_step=self.global_step)

        # Forward pass
        pred_logits, target_tokens = self.forward(
            batch.audio,
            batch.map_features,
            batch.tokens,
            batch.timestamps,
            batch.valid,
        )
        
        # Calculate loss
        loss = self.criterion(pred_logits.reshape(-1, pred_logits.size(-1)), target_tokens.reshape(-1))
        
        # Calculate accuracy
        pred_tokens = pred_logits.argmax(dim=-1)
        accuracy = (pred_tokens == target_tokens).float().mean()
        
        # Log metrics
        self.log('val/loss', loss, batch_size=batch.tokens.size(0))
        self.log('val/accuracy', accuracy, batch_size=batch.tokens.size(0))

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
    
    @th.no_grad()
    def sample(
        self,
        audio: Float[Tensor, "A L"],
        map_features: Float[Tensor, "M"],
        max_len: int = -1,
        top_p: float = .9,
        show_progress: bool = False,
    ) -> Int[Tensor, "1 N"]:
        self.eval()

        bos_id = self.vocab.ids[Token(TokenType.BOS)]
        eos_id = self.vocab.ids[Token(TokenType.EOS)]

        # precompute audio features
        audio_features = self.audio_encoder(audio[None])
        global_ctx = self.global_encoder(audio_features)
        multiscale_features = self.ctx_encoder.precompute(audio_features)
        frame_times = th.tensor(get_frame_times(audio_features.size(-1)), device=audio.device)
        audio_duration_ms = frame_times[-1].item()

        # initialize sequence and time
        token_id = bos_id
        current_time_ms = 0
        generated_tokens = [bos_id]
        
        # initialize kv cache
        cache = None

        # initialize logit processor
        lp = LogitProcessor(self.vocab)
        logit_mask = th.tensor(lp.advance(bos_id), device=audio.device)

        with tqdm(total=int(audio_duration_ms), desc="sampling", disable=not show_progress) as pbar:
            for i in itertools.count():
                if max_len > 0 and i >= max_len:
                    break
                
                if current_time_ms > audio_duration_ms and logit_mask[eos_id]:
                    generated_tokens.append(eos_id)
                    break

                # prepare inputs for single-token forward pass
                token_tensor = th.tensor([[token_id]], device=self.device, dtype=th.long)
                timestamp_tensor = th.tensor([[current_time_ms]], device=self.device, dtype=th.float32)

                # get context for the current token
                frame_idxs = th.searchsorted(frame_times, timestamp_tensor)
                multi_scale_ctx = self.ctx_encoder(multiscale_features, frame_idxs)
                
                expanded_global_ctx = global_ctx[None, None, ...].expand(1, 1, -1, -1)
                ctx = th.cat([expanded_global_ctx, multi_scale_ctx], dim=2)

                embs = self.token_embed(token_tensor)

                # decode one step
                output, cache = self.decoder(embs, ctx=ctx, cache=cache)
                logits = self.token_head(output)

                # apply logit mask
                logits = th.where(logit_mask, logits, -th.inf)

                # sample next token
                if top_p <= 0:
                    next_token_id = int(th.argmax(logits, dim=-1).item())
                else:
                    # Nucleus (top-p) sampling
                    probs = F.softmax(logits, dim=-1)
                    sorted_probs, sorted_indices = th.sort(probs, descending=True)
                    cumulative_probs = th.cumsum(sorted_probs, dim=-1)
                    # Find cutoff
                    cutoff = (cumulative_probs > top_p).float().argmax().item() + 1
                    top_p_probs = sorted_probs[..., :cutoff]
                    top_p_indices = sorted_indices[..., :cutoff]
                    # Renormalize
                    top_p_probs = top_p_probs / top_p_probs.sum(dim=-1, keepdim=True)
                    sampled_idx = th.multinomial(top_p_probs.squeeze(0), num_samples=1)
                    next_token_id = int(top_p_indices.squeeze(0)[sampled_idx].item())

                generated_tokens.append(next_token_id)

                # stop if EOS
                if next_token_id == eos_id:
                    break

                token_id = next_token_id
                logit_mask = th.tensor(lp.advance(token_id), device=audio.device)

                # update time
                last_time_ms = current_time_ms
                token = self.vocab.tokens[token_id]
                if token.typ == TokenType.TIME_SHIFT:
                    current_time_ms += self.vocab.time_shifts[token.value]
                
                pbar.update(current_time_ms - last_time_ms)
        
        return th.tensor([generated_tokens], device=self.device, dtype=th.long)