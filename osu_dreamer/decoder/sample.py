
from jaxtyping import Float, Shaped, Int

import time
import tqdm

import torch as th
from torch import Tensor
import torch.nn.functional as F

from osu_dreamer.data.labels import NUM_LABELS
from osu_dreamer.data.load_audio import A_DIM, get_frame_times

from .data.tokens import Token, decode, PAD, BOS, EOS, DIFF, T0

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
) -> tuple[
    list[list[Token|float]],                # list of B lists of tokens and timestamps
    Float[Tensor, str(f"B {NUM_LABELS}")],  # predicted labels
]:
    D = audio.device
    end_time = time.time() + time_budget
    
    labels[:,0] = -1 # zero out sr label
    c = model.label_emb(labels) # B C
    B = c.size(0)
    pred_labels = [ [] for _ in range(B) ]
    
    L = audio.size(-1)
    half_window = int(model.seq_len * .5)
    ctx = model.audio_encoder(F.pad(audio, (0, model.seq_len-1))[None])[0].transpose(0,1) # L+s-1 H
    frame_times = th.tensor(get_frame_times(L+model.seq_len-1))

    prelude_tokens = th.tensor([DIFF, BOS], device=D).long().repeat(B,1)

    active_batches = th.arange(B, device=D)             # samples that are still generating
    cur_start_idxs = th.zeros(B, device=D).long()       # current context window start index
    cur_tail_idx = th.zeros(B, device=D).long()         # token index @ end of generation
    cur_tokens = th.empty(B,0, device=D).long()         # tokens currently in generation
    cur_token_idxs = th.empty(B,0, device=D).long()     # token positioning

    output_tokens: list[list[Token | float]] = [ [] for _ in range(B) ]

    pbar = tqdm.tqdm(total=L*B, disable=not show_progress)
    while True:
        if time.time() > end_time:
            # time limit reached
            break

        # index into (B,T)-shaped tensors to get values corresponding to most recent generation
        TAIL = (th.arange(B), cur_tail_idx)

        cur_ctx_idxs = th.arange(model.seq_len, device=D)[None] + cur_start_idxs[:,None]
        cur_ctx = ctx[cur_ctx_idxs] # B bL H

        prelude_token_idxs = cur_ctx_idxs[:,:1].repeat(1, 2)
        h = model.decoder(
            x = model.embed(th.cat([prelude_tokens, cur_tokens], dim=1)),
            x_t = th.cat([prelude_token_idxs, cur_token_idxs], dim=1),
            ctx = cur_ctx, 
            ctx_t = cur_ctx_idxs, 
            c = c,
        ) # B n+2 E

        # predict labels
        label_emb = h[:,0] # B E
        for i, pred_label in zip(active_batches, model.label_head(label_emb)):
            pred_labels[i].append(pred_label)

        # predict tokens
        pred_embs = h[:,1:] # B n+1 E
        pred_token_logits = model.token_head(pred_embs[TAIL]) # B V

        # latest timing generated so far (no timing yet = ctx start)
        cur_latest_token_idxs = th.cat([ cur_start_idxs[:,None], cur_token_idxs ], dim=1)[TAIL] # B

        # disallow timing into the past
        for b_idx, num_mask in enumerate(cur_latest_token_idxs - cur_start_idxs):
            pred_token_logits[b_idx, T0:T0+num_mask] = -th.inf

        pred_tokens = th.multinomial(pred_token_logits.softmax(dim=-1), num_samples=1)[:,0] # B
        pred_token_idxs = th.where(
            pred_tokens >= T0,                      # timing predicted
            cur_start_idxs + pred_tokens - T0,      # ? use index from predicted timing
            cur_latest_token_idxs,                  # : use index from previously latest timing
        )

        # update windows
        update = th.where(
            pred_tokens == EOS,                                         # no tokens remaining for current window
            model.seq_len,                                               # ? go to next window
            th.where(
                pred_tokens >= T0,                                      # new timestamp predicted
                th.clamp((pred_tokens - T0) - half_window, min=0),      # ? slide forward until half of window is future context
                0,                                                      # : no window updates
            ),
        )
        pbar.update(update.sum().item())
        cur_start_idxs += update

        # grow sequence
        if (cur_tail_idx >= cur_tokens.size(1)).any():
            cur_tokens = th.cat([cur_tokens, th.full((B, 1), PAD, device=D)], dim=1)
            cur_token_idxs = th.cat([cur_token_idxs, th.full((B, 1), -1, device=D)], dim=1)

        # update sequence
        pred_tokens[pred_tokens == EOS] = PAD
        cur_tokens[TAIL] = pred_tokens
        cur_token_idxs[TAIL] = pred_token_idxs
        cur_tail_idx += (pred_tokens != PAD).long()
        del TAIL

        # dequeue tokens that are before start of new window
        shifts = (cur_token_idxs >= cur_start_idxs[:,None]).long().argmax(dim=1) # B
        cur_tokens = roll_by_shifts(cur_tokens, shifts)
        cur_token_idxs = roll_by_shifts(cur_token_idxs, shifts)
        cur_tail_idx -= shifts

        for b, (shift,batch_idx) in enumerate(zip(shifts, active_batches)):
            sl = (b, slice(cur_tokens.size(1)-shift,None)) # ...[b,-shift:]

            output_tokens[batch_idx].extend((
                decode(int(token)) if token < T0 else float(frame_times[token_idx])
                for token, token_idx in zip(cur_tokens[sl], cur_token_idxs[sl])
            ))
            cur_tokens[sl] = PAD
            cur_token_idxs[sl] = -1

        # remove completed samples from batch 
        next_active = cur_start_idxs < L
        if not next_active.any():
            # all completed
            break

        if not next_active.all():
            prelude_tokens = prelude_tokens[next_active]
            active_batches = active_batches[next_active]
            cur_start_idxs = cur_start_idxs[next_active]
            cur_tail_idx = cur_tail_idx[next_active]
            cur_tokens = cur_tokens[next_active]
            cur_token_idxs = cur_token_idxs[next_active]
            c = c[next_active]
            B = active_batches.size(0)

        # shrink sequence
        while cur_tokens.size(1) > 0 and (cur_tokens[:,-1] == PAD).all():
            cur_tokens = cur_tokens[:,:-1]
            cur_token_idxs = cur_token_idxs[:,:-1]
            
    pbar.close()

    for b, (tail_idx,batch_idx) in enumerate(zip(cur_tail_idx, active_batches)):
        sl = (b, slice(tail_idx)) # ...[b,:tail_idx]
        output_tokens[batch_idx].extend((
            decode(int(token)) if token < T0 else float(frame_times[token_idx])
            for token, token_idx in zip(cur_tokens[sl], cur_token_idxs[sl])
        ))

    return output_tokens, th.stack([ th.stack(l).mean(dim=0) for l in pred_labels ])