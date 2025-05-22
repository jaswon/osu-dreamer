
from jaxtyping import Float, Int

import torch as th
from torch import Tensor

from osu_dreamer.data.load_audio import get_frame_times

from .data.tokens import PAD, EOS

@th.no_grad
def make_batch(
    types: Int[Tensor, "N"],
    tokens: Int[Tensor, "N"],
    timestamps: Float[Tensor, "N"],
    positions: Float[Tensor, "N 2"],
    seq_len: int,
    num_frames: int,
    max_token_numel: int,
) -> tuple[
    Int[Tensor, "B S"],         # audio timing 
    Int[Tensor, "B T"],         # output types
    Int[Tensor, "B T"],         # token timing
    Int[Tensor, "B T"],         # tokens 
    Float[Tensor, "B T 2"],     # token positioning
]:
    D = tokens.device
    frame_times = th.tensor(get_frame_times(num_frames), device=D).float() # L
    token_frame_idxs = th.searchsorted(frame_times, timestamps)

    max_tokens = 0
    batches: list[tuple[int, int, int]] = []
    for ctx_start_idx in th.randperm(num_frames - seq_len).tolist():
        token_start_idx, token_mid_idx, token_end_idx = th.searchsorted(
            token_frame_idxs,
            th.tensor([
                ctx_start_idx, 
                ctx_start_idx+int(seq_len/2), 
                ctx_start_idx+seq_len,
            ], device=D),
        ).tolist()

        # only include tokens one timestamp past half context
        if token_mid_idx < len(timestamps):
            token_end_idx = int(th.clamp(
                th.argmax((timestamps > timestamps[token_mid_idx]).long()),
                min = token_mid_idx,
                max = token_end_idx,
            ).item())

        num_tokens = token_end_idx - token_start_idx
        if (1+max(max_tokens, num_tokens)) * (len(batches)+1) > max_token_numel:
            if len(batches) == 0:
                continue
            break
        max_tokens = max(max_tokens, num_tokens)
        batches.append((ctx_start_idx, token_start_idx, token_end_idx))

    b_ctx_idxs = th.empty(len(batches), seq_len).long().to(D) # B S
    b_types = th.full((len(batches), max_tokens+1), 0).long().to(D)
    b_tokens = th.full((len(batches), max_tokens+1), PAD).to(D)
    b_token_idxs = th.full((len(batches), max_tokens+1), -1).long().to(D)
    b_token_locs = th.full((len(batches), max_tokens+1, 2), 0.).to(D)

    for idx, (ctx_start_idx, token_start_idx, token_end_idx) in enumerate(batches):
        b_ctx_idxs[idx] = th.arange(ctx_start_idx,ctx_start_idx+seq_len)

        num_tokens = token_end_idx - token_start_idx
        b_types[idx, :num_tokens] = types[token_start_idx:token_end_idx]
        b_token_idxs[idx, :num_tokens] = token_frame_idxs[token_start_idx:token_end_idx]
        b_token_locs[idx, :num_tokens] = positions[token_start_idx:token_end_idx]
        b_tokens[idx, :num_tokens] = tokens[token_start_idx:token_end_idx]
        b_tokens[idx, num_tokens] = EOS

    return b_ctx_idxs, b_types, b_token_idxs, b_tokens, b_token_locs