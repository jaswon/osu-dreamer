from typing import Any
from jaxtyping import Float, Int

import torch as th
from torch import Tensor

import pytorch_lightning as pl
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import itertools

from osu_dreamer.lm.data.tokens.state import LogitProcessor
from osu_dreamer.modules.muon import Muon
from osu_dreamer.modules.lr_schedule import LRScheduleArgs, make_lr_schedule
from osu_dreamer.data.load_audio import MS_PER_FRAME

from .data.dataset import Batch
from .data.tokens.tokens import Vocab, TokenType

from .modules.audio_encoder import AudioEncoder, AudioEncoderArgs
from .modules.decoder import Decoder, DecoderArgs
from .modules.head.head import DecoderHead


class Model(pl.LightningModule):
    def __init__(
        self,

        # training parameters
        opt_args: dict[str, Any],
        schedule_args: LRScheduleArgs,
        timing_jitter: int, 

        # sampling parameters
        context_shift_threshold: float,
        context_shift_amount: float,
        
        # model hparams
        vocab: Vocab,       # set from config.data
        context_size: int,  # set from config.data
        emb_dim: int,
        head_h_dim: int,
        decoder_args: DecoderArgs,
        
        # audio encoder hparams
        ctx_dim: int,
        audio_encoder_args: AudioEncoderArgs,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # training params
        self.opt_args = opt_args
        self.lr_schedule = make_lr_schedule(schedule_args)
        self.timing_jitter = timing_jitter

        # sampling params
        assert 0 < context_shift_threshold < 1
        assert 0 < context_shift_amount < 1
        self.context_shift_threshold = context_shift_threshold
        self.context_shift_amount = context_shift_amount
        
        # model components
        self.vocab = vocab
        self.context_size = context_size
        self.decoder = Decoder(emb_dim, ctx_dim, decoder_args, context_size)
        self.head = DecoderHead(vocab, emb_dim, head_h_dim)
        
        # audio encoder
        self.audio_encoder = AudioEncoder(ctx_dim, audio_encoder_args, vocab.time_bins)
        
    
    def forward(
        self,
        map_features: Float[Tensor, "B M"],
        audio: Float[Tensor, "B A L"],
        seq: Int[Tensor, "B Np1 4"],
        validation: bool = False,
        input_jitter: bool = True,
    ) -> tuple[
        Float[Tensor, ""],  # loss
        dict[str, float],   # log dict
    ]:
        ctx = self.audio_encoder(audio) # B L D

        inp = seq[:,:-1]

        if input_jitter:
            # jitter input time tokens to improve timing robustness
            timing_jitter = th.randint_like(inp[:,:,3], -self.timing_jitter, self.timing_jitter+1)
            inp[:,:,3] = th.clamp(inp[:,:,3] + timing_jitter, min=0, max=self.vocab.time_bins)

        output, _ = self.decoder(self.head.embed(inp), ctx=ctx)
        return self.head(output, seq[:,1:], validation)
    
    def training_step(self, batch: Batch, batch_idx: int):
        # Forward pass
        loss, log_dict = self.forward(
            batch.map_features,
            batch.audio,
            batch.seq,
        )
        self.log_dict({ f"train/{k}": v for k,v in log_dict.items() })
        return loss
    
    def validation_step(self, batch: Batch, batch_idx: int):
        
        # On the first validation batch of every epoch, generate a sample
        if batch_idx == 0 and self.global_rank == 0:
            seq = self.sample(batch.audio[0], batch.map_features[0], max_len=512, top_p=0.)
            generated_tokens = [
                (
                    f"{MS_PER_FRAME * t / 1000:.3f}" if i == self.vocab.TIME else
                    f"({x},{y})" if i == self.vocab.POS else
                    self.vocab.tokens[i]
                )
                for i,x,y,t in seq
            ]

            exp: SummaryWriter = self.logger.experiment # type: ignore
            sample_text = '\n'.join([ str(event) for event in generated_tokens ])
            exp.add_text(f'sample', sample_text, global_step=self.global_step)

        # Forward pass
        loss, log_dict = self.forward(
            batch.map_features,
            batch.audio,
            batch.seq,
            validation=True,
        )
        self.log_dict({ f"val/{k}": v for k,v in log_dict.items() })
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
    ) -> list[tuple[int, int, int, int]]:
        self.eval()

        # precompute sampling params
        ctx_size = self.vocab.time_bins
        L = audio.size(-1)
        max_ctx_start = L - ctx_size
        ctx_shift_threshold = round(self.context_shift_threshold * ctx_size)
        ctx_shift_amount = round(self.context_shift_amount * ctx_size)

        # initialize sequence
        cur_id = self.vocab.BOS
        cur_x = 256
        cur_y = 192
        cur_t = 0
        seq: list[tuple[int, int, int, int]] = [(cur_id, cur_x, cur_y, cur_t)]

        ctx_start = 0
        ctx_starts = [ctx_start]
        ctx = None
        cache = None

        # initialize logit processor
        lp = LogitProcessor(self.vocab)
        logit_mask = th.tensor(lp.advance(cur_id), device=self.device)

        with tqdm(total=int(audio.size(-1)), desc="sampling", disable=not show_progress) as pbar:
            for i in itertools.count():

                # limit generation
                if max_len > 0 and i >= max_len:
                    break

                # decode one step
                if ctx is None:
                    ctx = self.audio_encoder(audio[None,:,ctx_start:ctx_start+ctx_size])
                
                # If cache has been invalidated, do full forward pass with context-adjusted tokens
                if cache is None:
                    seq_tensor = self._get_seq_relative_to_ctx_start(seq, ctx_start) # 1 N 4
                else:
                    seq_tensor = th.tensor(seq[-1], device=self.device, dtype=th.long)[None,None] # 1 1 4

                pred_embs, cache = self.decoder(self.head.embed(seq_tensor), ctx=ctx, cache=cache)
                sample = self.head.sample(pred_embs[0,-1], logit_mask, cur_t, top_p)
                cur_id, sample_x, sample_y, sample_t = tuple[int,int,int,int](sample.tolist())

                commit_sample = True
                if cur_id == self.vocab.POS:
                    cur_x, cur_y = sample_x, sample_y

                elif cur_id == self.vocab.TIME:

                    advance_context = False
                    if sample_t == self.vocab.time_bins:
                        # eos - advance without committing
                        advance_context = True
                        commit_sample = False
                    else:
                        cur_t = sample_t

                    if cur_t > ctx_shift_threshold:
                        # past shift threshold - advance
                        advance_context = True

                        pbar_step = ctx_start + cur_t - pbar.n
                        if pbar_step > 0:
                            pbar.update(pbar_step)

                    if advance_context:
                        # check if we're done
                        if ctx_start >= max_ctx_start:
                            # check if incomplete event
                            match self.vocab.tokens[seq[-1][0]].typ:
                                case TokenType.SLIDER | TokenType.SPINNER | TokenType.BREAK:
                                    # pop start of incomplete event
                                    while True:
                                        seq.pop()
                                        ctx_starts.pop()
                                        if seq[-1][0] == self.vocab.TIME:
                                            seq.pop()
                                            ctx_starts.pop()
                                            break
                            break

                        # shift context forward, and invalidate cache
                        shift_size = min(ctx_shift_amount, L - ctx_size - ctx_start)
                        ctx_start += shift_size
                        cur_t -= shift_size
                        cache = None
                        ctx = None

                if commit_sample:
                    seq.append((cur_id, cur_x, cur_y, cur_t))
                    ctx_starts.append(ctx_start)
                    logit_mask = th.tensor(lp.advance(cur_id), device=self.device)
                ctx_starts[-1] = ctx_start

        seq.append((self.vocab.EOS, cur_x, cur_y, cur_t))
        ctx_starts.append(ctx_start)

        return [
            (i,x,y,ctx_start+t)
            for (i,x,y,t), ctx_start in zip(seq, ctx_starts)
        ]
    
    
    def _get_seq_relative_to_ctx_start(
        self, 
        seq: list[tuple[int, int, int, int]], 
        ctx_start: int, 
    ) -> Int[Tensor, "1 N 4"]:
        """
        Adjust TIME tokens in a sequence to be relative to a new context start.
        """
        new_seq = []
        temp_buffer = []
        
        # Process tokens in reverse order
        for i in range(len(new_seq) - 1, -1, -1):
            i_id, i_x, i_y, i_t = seq[i]
            token = self.vocab.tokens[i_id]
            
            if token.typ != TokenType.TIME:
                # collect non-TIME tokens in a temp buffer
                temp_buffer.append((i_id, i_x, i_y, i_t - ctx_start))
            else:
                # Hit a TIME token - evaluate the complete event we've collected
                time_bin = i_t - ctx_start
                
                if 0 <= time_bin < self.vocab.time_bins:
                    # time is valid- include them in return
                    new_seq.extend(temp_buffer)
                    new_seq.append((i_id, i_x, i_y, time_bin))
                    temp_buffer = []

                    if len(new_seq) >= self.context_size:
                        new_seq = new_seq[:self.context_size]
                        break
                else:
                    # TIME token before context window, exit early
                    break

        if len(new_seq) == 0:
            # no history, return BOS
            new_seq = [(self.vocab.BOS, 256, 192, 0)]
                
        # Reverse the final result since we built it backwards
        return th.tensor(list(reversed(new_seq)), device=self.device, dtype=th.long)[None]