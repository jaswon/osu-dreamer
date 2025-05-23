
from jaxtyping import Float, Int

import random

import torch as th
from torch import Tensor

from osu_dreamer.data.load_audio import get_frame_times

from .data.tokens import PAD

@th.no_grad
def make_batch(
    modes: Int[Tensor, "N"],
    tokens: Int[Tensor, "N"],
    timestamps: Float[Tensor, "N"],
    positions: Float[Tensor, "N 2"],
    src_len: int,
    num_frames: int,
    batch_size: int,
    tgt_len: int,
) -> tuple[
    Int[Tensor, "B S"],         # ctx idxs
    Int[Tensor, "B T"],         # modes
    Int[Tensor, "B T"],         # tokens 
    Int[Tensor, "B T"],         # timings
    Float[Tensor, "B T 2"],     # positions
]:
    D = tokens.device
    frame_times = th.tensor(get_frame_times(num_frames), device=D).float() # L
    token_frame_idxs = th.searchsorted(frame_times, timestamps) # T

    b_ctx_idxs = th.empty(batch_size, src_len).long().to(D) # B S
    b_modes = th.full((batch_size, tgt_len), 0).long().to(D)
    b_tokens = th.full((batch_size, tgt_len), PAD).long().to(D)
    b_timings = th.full((batch_size, tgt_len), 0).long().to(D)
    b_positions = th.full((batch_size, tgt_len, 2), 0.).to(D)

    for idx, ctx_start_idx in enumerate(th.randperm(num_frames - src_len)[:batch_size].tolist()):
        token_start_idx, token_end_idx = th.searchsorted(
            token_frame_idxs,
            th.tensor([
                ctx_start_idx, 
                ctx_start_idx+src_len,
            ], device=D),
        ).tolist()
    
        b_ctx_idxs[idx] = th.arange(ctx_start_idx,ctx_start_idx+src_len)

        if token_end_idx - token_start_idx > tgt_len:
            token_start_idx = random.randint(token_start_idx, token_end_idx-tgt_len)
            token_end_idx = token_start_idx + tgt_len

        num_tokens = token_end_idx - token_start_idx
        sl = slice(token_start_idx,token_end_idx)
        b_modes[idx, :num_tokens] = modes[sl]
        b_tokens[idx, :num_tokens] = tokens[sl]
        b_timings[idx, :num_tokens] = token_frame_idxs[sl] - ctx_start_idx
        b_positions[idx, :num_tokens] = positions[sl]

    return b_ctx_idxs, b_modes, b_tokens, b_timings, b_positions