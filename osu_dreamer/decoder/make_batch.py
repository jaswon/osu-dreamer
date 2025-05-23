
from jaxtyping import Float, Int

import torch as th
from torch import Tensor

from osu_dreamer.data.load_audio import get_frame_times

from .data.tokens import PAD, EOS, DIFF, BOS

@th.no_grad
def make_batch(
    modes: Int[Tensor, "N"],
    tokens: Int[Tensor, "N"],
    timings: Float[Tensor, "N"],
    positions: Float[Tensor, "N 2"],
    seq_len: int,
    num_frames: int,
    max_token_numel: int,
) -> tuple[
    Int[Tensor, "B S"],         # audio timing
    Int[Tensor, "B T"],         # modes
    Int[Tensor, "B T"],         # tokens 
    Int[Tensor, "B T"],         # timings
    Float[Tensor, "B T 2"],     # positions
]:
    D = tokens.device
    frame_times = th.tensor(get_frame_times(num_frames), device=D).float() # L
    token_frame_idxs = th.searchsorted(frame_times, timings) # T

    max_tokens = 0
    batches: list[tuple[int, int, int]] = []
    for ctx_start_idx in th.randperm(num_frames - seq_len).tolist():
        token_start_idx, token_end_idx = th.searchsorted(
            token_frame_idxs,
            th.tensor([
                ctx_start_idx, 
                ctx_start_idx+seq_len,
            ], device=D),
        ).tolist()

        num_tokens = token_end_idx - token_start_idx
        if (1+max(max_tokens, num_tokens)) * (len(batches)+1) > max_token_numel:
            if len(batches) == 0:
                continue
            break
        max_tokens = max(max_tokens, num_tokens)
        batches.append((ctx_start_idx, token_start_idx, token_end_idx))

    b_ctx_idxs = th.empty(len(batches), seq_len).long().to(D) # B S
    b_modes = th.full((len(batches), max_tokens), 0).long().to(D)
    b_tokens = th.full((len(batches), max_tokens), PAD).long().to(D)
    b_timings = th.full((len(batches), max_tokens), 0).long().to(D)
    b_positions = th.full((len(batches), max_tokens, 2), 0.).to(D)
    
    for idx, (ctx_start_idx, token_start_idx, token_end_idx) in enumerate(batches):
        b_ctx_idxs[idx] = th.arange(ctx_start_idx,ctx_start_idx+seq_len)

        num_tokens = token_end_idx - token_start_idx
        sl = slice(token_start_idx,token_end_idx)
        b_modes[idx, :num_tokens] = modes[sl]
        b_tokens[idx, :num_tokens] = tokens[sl]
        b_timings[idx, :num_tokens] = token_frame_idxs[sl] - ctx_start_idx
        b_positions[idx, :num_tokens] = positions[sl]

    return b_ctx_idxs, b_modes, b_tokens, b_timings, b_positions