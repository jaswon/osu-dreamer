
from jaxtyping import Float, Shaped, Int

import time
import tqdm

import torch as th
from torch import Tensor
import torch.nn.functional as F

from osu_dreamer.data.labels import NUM_LABELS
from osu_dreamer.data.load_audio import A_DIM, get_frame_times

from .data.tokens import Token, decode, PAD, BOS, EOS, DIFF
from .data.tokenize import PositionToken, TimingToken

from .model import Model

def right_pad_dims_to(x: Tensor, t: Tensor):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def roll_by_shifts(
    inputs: Shaped[Tensor, "B N ..."], 
    shifts: Int[Tensor, "B"],
) -> Shaped[Tensor, "B N ..."]:
    b,n,*_ = inputs.size()
    col_idxs = th.arange(n, device=shifts.device).repeat(b,1)
    shifted_idxs = (col_idxs + shifts[:,None]) % n
    shifted_idxs = right_pad_dims_to(inputs, shifted_idxs).expand(-1,-1,*inputs.shape[2:])
    return th.gather(inputs, 1, shifted_idxs.long())

@th.no_grad
def sample(
    model: Model,
    audio: Float[Tensor, str(f"{A_DIM} L")],
    labels: Float[Tensor, str(f"B {NUM_LABELS}")],
    time_budget: int | float = float('inf'), # max allowed time (sec)
    show_progress: bool = False,
) -> list[list[Token | TimingToken | PositionToken]]: # list of B lists of tokens and timestamps
    D = audio.device
    end_time = time.time() + time_budget
    
    labels[:,0] = -1 # zero out sr label
    c = model.label_emb(labels) # B C
    B = c.size(0)
    
    L = audio.size(-1)
    half_window = int(model.seq_len * .5)
    ctx = model.audio_encoder(F.pad(audio, (0, model.seq_len-1))[None].transpose(1,2))[0] # L+S-1 H
    frame_times = th.tensor(get_frame_times(L+model.seq_len-1))

    active_batches = th.arange(B).to(D)             # samples that are still generating
    cur_start_idxs = th.zeros(B).long().to(D)       # current context window start index
    cur_tail_idx = th.zeros(B).long().to(D)         # index of most recently generated token

    cur_idxs        = th.tensor([0]).to(D).repeat(B,1)
    cur_modes       = th.tensor([0]).to(D).repeat(B,1)          # modes
    cur_tokens      = th.tensor([BOS]).to(D).repeat(B,1)        # tokens
    cur_positions   = th.tensor([[0.,0.]]).to(D).repeat(B,1,1)  # positions

    def pred_output(sl):
        for typ, token, token_idx, (x,y) in zip(
            cur_modes[sl], 
            cur_tokens[sl], 
            cur_idxs[sl],
            cur_positions[sl],
        ):
            yield [
                decode(int(token)),
                TimingToken(float(frame_times[token_idx])),
                PositionToken((x.item()+4)*64,(y.item()+3)*64),
            ][typ]

    output_tokens: list[list[Token | TimingToken | PositionToken]] = [ [] for _ in range(B) ]

    pbar = tqdm.tqdm(total=L*B, disable=not show_progress)
    while True:
        if time.time() > end_time:
            # time limit reached
            break

        B = active_batches.size(0)
        rB = th.arange(B)
        cur_ctx_idxs = th.arange(model.seq_len).to(D)[None] + cur_start_idxs[:,None] # B S
        cur_ctx = ctx[cur_ctx_idxs] # B S H

        cur_timings = cur_idxs - cur_start_idxs[:,None]

        cur_pad_idxs = th.arange(cur_modes.size(1)).to(D) > cur_tail_idx[:,None]
        cur_embs = model.modal_head.embed(
            modes = cur_modes.index_put((cur_pad_idxs,), th.tensor(0)),
            tokens = cur_tokens.index_put((cur_pad_idxs,), th.tensor(PAD)),
            timings = cur_timings.index_put((cur_pad_idxs,), th.tensor(0)),
            positions = cur_positions.index_put((cur_pad_idxs,), th.tensor(0)),
        )

        h = model.decoder(
            x = cur_embs,
            x_t = cur_idxs,
            ctx = cur_ctx, 
            ctx_t = cur_ctx_idxs, 
            c = c,
        ) # B n E

        ( # predict token types
            pred_modes, 
            pred_tokens, 
            pred_timings, 
            pred_positions,
        ) = model.modal_head.sample(
            embs = h[rB, cur_tail_idx], 
            timings_to_mask = cur_timings[rB, cur_tail_idx], # prevent emitting already emitted timings
        )

        pred_latest_idxs = th.where(
            pred_modes == 1,                    # timing predicted
            pred_timings + cur_start_idxs,      # ? use predicted timings
            cur_idxs[rB, cur_tail_idx],         # : use current timings
        ) # B

        # grow sequence
        cur_idxs = th.cat([cur_idxs, cur_idxs[:,-1:]], dim=1)
        cur_modes = th.cat([cur_modes, cur_modes[:,-1:]], dim=1)
        cur_tokens = th.cat([cur_tokens, cur_tokens[:,-1:]], dim=1)
        cur_positions = th.cat([cur_positions, cur_positions[:,-1:]], dim=1)

        # update sequence
        cur_tail_idx += ((pred_modes != 0) | (pred_tokens != PAD)).long()
        pred_tokens[pred_tokens == EOS] = PAD
        cur_idxs[rB, cur_tail_idx] = pred_latest_idxs
        cur_modes[rB, cur_tail_idx] = pred_modes
        cur_tokens[rB, cur_tail_idx] = pred_tokens
        cur_positions[rB, cur_tail_idx] = pred_positions

        # update windows
        pred_latest_timings = pred_latest_idxs - cur_start_idxs
        update = th.where(
            (pred_modes == 0) & (pred_tokens == EOS),               # no tokens remaining for current window
            model.seq_len,                                          # ? slide full window
            th.clamp(pred_latest_timings - half_window, min=0),     # : slide until half of window is future context
        ) # B
        pbar.update(update.sum().item())
        cur_start_idxs += update

        # dequeue tokens that are before start of new window
        shifts = (cur_timings >= 0).long().argmax(dim=1) # B
        cur_idxs = roll_by_shifts(cur_idxs, shifts)
        cur_modes = roll_by_shifts(cur_modes, shifts)
        cur_tokens = roll_by_shifts(cur_tokens, shifts)
        cur_positions = roll_by_shifts(cur_positions, shifts)
        cur_tail_idx -= shifts

        for b, (shift,batch_idx) in enumerate(zip(shifts, active_batches)):
            sl = (b, slice(cur_tokens.size(1)-shift,None)) # ...[b,-shift:]
            output_tokens[batch_idx].extend(pred_output(sl))

        # remove completed samples from batch 
        next_active = cur_start_idxs < L
        if not next_active.any():
            # all completed
            break

        if not next_active.all():
            active_batches = active_batches[next_active]
            cur_start_idxs = cur_start_idxs[next_active]
            cur_tail_idx = cur_tail_idx[next_active]
            cur_idxs = cur_idxs[next_active]
            cur_modes = cur_modes[next_active]
            cur_tokens = cur_tokens[next_active]
            cur_positions = cur_positions[next_active]
            c = c[next_active]

        # shrink sequence
        while cur_tokens.size(1) > 0 and ((cur_modes[:,-1] == 0) & (cur_tokens[:,-1] == PAD)).all():
            cur_idxs = cur_idxs[:,:-1]
            cur_modes = cur_modes[:,:-1]
            cur_tokens = cur_tokens[:,:-1]
            cur_positions = cur_positions[:,:-1]
            
    pbar.close()

    for b, (tail_idx,batch_idx) in enumerate(zip(cur_tail_idx, active_batches)):
        sl = (b, slice(tail_idx)) # ...[b,:tail_idx]
        output_tokens[batch_idx].extend(pred_output(sl))

    return output_tokens