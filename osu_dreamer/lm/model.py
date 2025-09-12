from typing import Any
from jaxtyping import Float, Int

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
from osu_dreamer.data.load_audio import MS_PER_FRAME, get_frame_times

from .data.dataset import Batch
from .data.tokens.tokens import Vocab, Token, TokenType

from .modules.audio_encoder import SimpleAudioEncoder
from .modules.decoder import Decoder, DecoderArgs


class Model(pl.LightningModule):
    def __init__(
        self,

        # training parameters
        opt_args: dict[str, Any],
        schedule_args: LRScheduleArgs,

        # sampling parameters
        context_shift_threshold: float,
        context_shift_amount: float,
        
        # model hparams
        vocab: Vocab,
        emb_dim: int,
        decoder_args: DecoderArgs,
        
        # audio encoder hparams
        ctx_dim: int,
        audio_h_dim: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # training params
        self.opt_args = opt_args
        self.lr_schedule = make_lr_schedule(schedule_args)

        # sampling params
        assert 0 < context_shift_threshold < 1
        assert 0 < context_shift_amount < 1
        self.context_shift_threshold = context_shift_threshold
        self.context_shift_amount = context_shift_amount
        
        # model components
        self.vocab = vocab
        vocab_size = len(vocab.tokens)
        self.token_embed = nn.Embedding(vocab_size, emb_dim)
        self.decoder = Decoder(emb_dim, ctx_dim, decoder_args)
        self.token_head = nn.Linear(emb_dim, vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # 0 = PAD token
        
        # audio encoder
        self.audio_encoder = SimpleAudioEncoder(audio_h_dim)
        
    
    def forward(
        self,
        map_features: Float[Tensor, "M"],
        audio: Float[Tensor, "B A L"],
        tokens: Int[Tensor, "B Np1"],
    ) -> tuple[
        Float[Tensor, "B N V"], # pred logits
        Int[Tensor, "B N"]      # true targets
    ]:
        ctx = self.audio_encoder(audio) # B L D
        embs = self.token_embed(tokens[:,:-1]) # B N D
        output, _ = self.decoder(embs, ctx=ctx)
        logits = self.token_head(output) # B N V
        
        return logits, tokens[:,1:]
    
    def training_step(self, batch: Batch, batch_idx: int):
        # Forward pass
        pred_logits, target_tokens = self.forward(
            batch.map_features,
            batch.audio,
            batch.tokens,
        )
        
        loss = self.criterion(pred_logits.reshape(-1, pred_logits.size(-1)), target_tokens.reshape(-1))
        
        # Log metrics
        self.log('train/loss', loss, batch_size=batch.tokens.size(0))
        
        return loss
    
    def validation_step(self, batch: Batch, batch_idx: int):
        
        # On the first validation batch of every epoch, generate a sample
        if batch_idx == 0 and self.global_rank == 0:
            token_ids, ctx_starts = self.sample(batch.audio[0], batch.map_features, max_len=512, top_p=0.)
            generated_tokens = [
                tok if tok.typ != TokenType.TIME else f"{MS_PER_FRAME * (tok.value + ctx_start) / 1000:.3f}"
                for tid, ctx_start in zip(token_ids, ctx_starts) 
                for tok in [self.vocab.tokens[int(tid.item())]]
            ]

            exp: SummaryWriter = self.logger.experiment # type: ignore
            sample_text = '\n'.join([ str(event) for event in generated_tokens ])
            exp.add_text(f'sample', sample_text, global_step=self.global_step)

        # Forward pass
        pred_logits, target_tokens = self.forward(
            batch.map_features,
            batch.audio,
            batch.tokens,
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
    ) -> tuple[
        Int[Tensor, "N"], # token ids
        Int[Tensor, "N"], # ctx starts
    ]:
        self.eval()

        # precompute audio features
        L = audio.size(-1)
        audio_features = self.audio_encoder(audio[None]) # 1 l h
        frame_times = th.tensor(get_frame_times(L), device=audio.device)
        audio_duration_ms = frame_times[-1].item()

        # precompute sampling params
        ctx_size = self.vocab.time_bins
        max_ctx_start = L - ctx_size
        ctx_shift_threshold = round(self.context_shift_threshold * ctx_size)
        ctx_shift_amount = round(self.context_shift_amount * ctx_size)

        # initialize sequence
        bos_id = self.vocab.ids[Token(TokenType.BOS)]
        token_id = bos_id
        generated_tokens: list[int] = []
        ctx_starts: list[int] = []
        ctx_start = 0
        cur_frame = 0
        
        # initialize kv cache
        cache = None

        # initialize logit processor
        lp = LogitProcessor(self.vocab)

        with tqdm(total=int(audio_duration_ms), desc="sampling", disable=not show_progress) as pbar:
            for i in itertools.count():
                ctx_starts.append(ctx_start)
                generated_tokens.append(token_id)
                logit_mask = th.tensor(lp.advance(token_id), device=audio.device)

                # limit generation
                if max_len > 0 and i >= max_len:
                    break

                # decode one step
                token_tensor = th.tensor([[token_id]], device=self.device, dtype=th.long)
                ctx = audio_features[:,ctx_start:ctx_start+ctx_size]
                embs = self.token_embed(token_tensor)
                
                # If cache has been invalidated, do full forward pass with context-adjusted tokens
                if cache is None:
                    # Create adjusted token sequence where TIME tokens are translated to current context
                    adjusted_tokens = self._get_time_shifted_tokens(generated_tokens, ctx_starts)

                    if len(adjusted_tokens) == 0:
                        # no token history, set to bos
                        adjusted_tokens = [bos_id]
                    
                    # Do full forward pass with adjusted tokens
                    all_tokens = th.tensor([adjusted_tokens], device=self.device, dtype=th.long)
                    all_embs = self.token_embed(all_tokens)
                    all_output, cache = self.decoder(all_embs, ctx=ctx, cache=None)
                    logits = self.token_head(all_output)[0,-1] # Get logits for the last token
                else:
                    # Normal incremental decode
                    output, cache = self.decoder(embs, ctx=ctx, cache=cache)
                    logits = self.token_head(output)[0,0] # V

                # mask grammatically incorrect tokens
                logits = th.where(logit_mask, logits, -th.inf)
                
                # mask past time tokens
                for token_idx, token in enumerate(self.vocab.tokens):
                    if token.typ == TokenType.TIME and token.value < cur_frame:
                        logits[token_idx] = -th.inf

                # sample next token
                if top_p <= 0:
                    # Greedy sampling
                    sampled = th.argmax(logits, dim=-1)
                else:
                    # Nucleus sampling
                    sorted_probs, sorted_indices = th.sort(F.softmax(logits, dim=-1), descending=True)
                    cutoff = (th.cumsum(sorted_probs, dim=-1) > top_p).float().argmax().item() + 1
                    sampled_idx = th.multinomial(sorted_probs[:cutoff], num_samples=1)[0]
                    sampled = sorted_indices[sampled_idx]

                token_id = int(sampled.item())
                token = self.vocab.tokens[token_id]

                # context should be advanced if
                # - emitted TIME is past context shift threshold
                # - emitted EOS and context remains
                advance_context = False

                if token.typ == TokenType.EOS:
                    # check if we're done
                    if ctx_start >= max_ctx_start:
                        break

                    advance_context = True
                    
                elif token.typ == TokenType.TIME:
                    cur_frame = token.value
                    if cur_frame > ctx_shift_threshold:
                        advance_context = True

                    cur_time = MS_PER_FRAME * (ctx_start + cur_frame)
                    if cur_time > pbar.n:
                        pbar.update(cur_time - pbar.n)

                if advance_context:
                    # shift context forward, and invalidate cache
                    shift_size = min(ctx_shift_amount, L - ctx_size - ctx_start)
                    ctx_start += shift_size
                    cur_frame -= shift_size
                    cache = None
        
        return (
            th.tensor(generated_tokens, device=self.device, dtype=th.long),
            th.tensor(ctx_starts, device=self.device, dtype=th.long),
        )
    
    def _get_time_shifted_tokens(
        self, 
        tokens: list[int], 
        ctx_starts: list[int], 
    ) -> list[int]:
        """
        Adjust TIME tokens in a sequence to be relative to a new context start.
        """
        adjusted = []
        temp_buffer = []
        
        # Process tokens in reverse order
        for i in range(len(tokens) - 1, -1, -1):
            token = self.vocab.tokens[tokens[i]]
            
            if token.typ != TokenType.TIME:
                # collect non-TIME tokens in a temp buffer
                temp_buffer.append(tokens[i])
            else:
                # Hit a TIME token - evaluate the complete event we've collected
                original_ctx = ctx_starts[i]
                new_time_bin = token.value + (original_ctx - ctx_starts[-1])
                
                if 0 <= new_time_bin < self.vocab.time_bins:
                    # time is valid- include them in return
                    adjusted.extend(temp_buffer)
                    adjusted.append(self.vocab.ids[Token(TokenType.TIME, new_time_bin)])
                    temp_buffer = []
                else:
                    # TIME token before context window, exit early
                    break
                
        # Reverse the final result since we built it backwards
        return list(reversed(adjusted))