
from jaxtyping import Float, Shaped, Int

import time
import tqdm

import torch as th
from torch import Tensor
import torch.nn.functional as F

from osu_dreamer.data.labels import NUM_LABELS
from osu_dreamer.data.load_audio import A_DIM, get_frame_times

from .data.tokens import Token, decode, PAD, BOS, EOS, DIFF

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
    ctx = model.audio_encoder(F.pad(audio, (0, model.seq_len-1))[None].transpose(1,2))[0] # L+s-1 H
    frame_times = th.tensor(get_frame_times(L+model.seq_len-1))

    prelude_token_embs = model.token_emb(th.tensor([DIFF, BOS], device=D).long()) # 2 E

    active_batches = th.arange(B).to(D)             # samples that are still generating
    cur_start_idxs = th.zeros(B).long().to(D)       # current context window start index
    cur_tail_idx = th.zeros(B).long().to(D)         # token index @ end of generation
    cur_types = th.empty(B,0).long().to(D)          # types of tokens
    cur_tokens = th.empty(B,0).long().to(D)         # discrete tokens
    cur_token_idxs = th.empty(B,0).long().to(D)     # token positioning (/ timing tokens)

    output_tokens: list[list[Token | float]] = [ [] for _ in range(B) ]

    pbar = tqdm.tqdm(total=L*B, disable=not show_progress)
    while True:
        if time.time() > end_time:
            # time limit reached
            break


        cur_ctx_idxs = th.arange(model.seq_len, device=D)[None] + cur_start_idxs[:,None] # B S
        cur_ctx = ctx[cur_ctx_idxs] # B bL H

        prelude_token_idxs = cur_start_idxs[:,None].repeat(1, 2)

        cur_token_embs = th.gather(
            th.stack([
                model.token_emb(cur_tokens),
                model.timestamp_emb(th.clamp(cur_token_idxs - cur_start_idxs[:,None], min=0)),
            ], dim=0), # K B N E
            dim = 0,
            index = cur_types[None,:,:,None].expand(-1,-1,-1,model.embed_dim),
        )[0]

        h = model.decoder(
            x = th.cat([prelude_token_embs.repeat(B,1,1), cur_token_embs], dim=1),
            x_t = th.cat([prelude_token_idxs, cur_token_idxs], dim=1),
            ctx = cur_ctx, 
            ctx_t = cur_ctx_idxs, 
            c = c,
        ) # B n+2 E

        # predict labels
        label_emb = h[:,0] # B E
        for i, pred_label in zip(active_batches, model.label_head(label_emb)):
            pred_labels[i].append(pred_label)

        # index into (B,T)-shaped tensors to get values corresponding to most recent generation
        TAIL = (th.arange(B), cur_tail_idx)

        # predict token types
        pred_embs = h[:,1:][TAIL] # B E
        pred_type_logits = model.type_head(pred_embs) # B 2
        pred_types = th.multinomial(pred_type_logits.softmax(dim=-1), num_samples=1)[:,0] # B

        # latest timing generated so far (no timing yet = ctx start)
        cur_latest_token_idxs = th.cat([ cur_start_idxs[:,None], cur_token_idxs ], dim=1)[TAIL] # B

        # predict timestamps, disallow timing into the past
        pred_timestamp_logits = model.timestamp_head(pred_embs) # B S
        for b_idx, num_mask in enumerate(cur_latest_token_idxs - cur_start_idxs):
            pred_timestamp_logits[b_idx, :num_mask] = -th.inf

        pred_timestamp = th.multinomial(pred_timestamp_logits.softmax(dim=-1), num_samples=1)[:,0] # B

        # predict tokens
        pred_token_logits = model.token_head(pred_embs) # B V
        pred_tokens = th.multinomial(pred_token_logits.softmax(dim=-1), num_samples=1)[:,0] # B

        pred_token_idxs = th.where(
            pred_types == 1,                        # timing predicted
            cur_start_idxs + pred_timestamp,        # ? use index from predicted timing
            cur_latest_token_idxs,                  # : use index from previously latest timing
        )

        # update windows
        update = th.where(
            pred_tokens == EOS,                                                 # no tokens remaining for current window
            model.seq_len,                                                      # ? slide full window
            th.clamp(pred_token_idxs - (cur_start_idxs + half_window), min=0),  # : slide until half of window is future context
        )
        pbar.update(update.sum().item())
        cur_start_idxs += update

        # grow sequence
        if (cur_tail_idx >= cur_tokens.size(1)).any():
            cur_types = th.cat([cur_types, th.full((B,1), 0).to(D)], dim=1)
            cur_tokens = th.cat([cur_tokens, th.full((B,1), PAD).to(D)], dim=1)
            cur_token_idxs = th.cat([cur_token_idxs, th.full((B,1), -1).to(D)], dim=1)

        # update sequence
        pred_tokens[pred_tokens == EOS] = PAD
        cur_types[TAIL] = pred_types
        cur_tokens[TAIL] = pred_tokens
        cur_token_idxs[TAIL] = pred_token_idxs
        cur_tail_idx += ((pred_types != 0) | (pred_tokens != PAD)).long()
        del TAIL

        # dequeue tokens that are before start of new window
        shifts = (cur_token_idxs >= cur_start_idxs[:,None]).long().argmax(dim=1) # B
        cur_types = roll_by_shifts(cur_types, shifts)
        cur_tokens = roll_by_shifts(cur_tokens, shifts)
        cur_token_idxs = roll_by_shifts(cur_token_idxs, shifts)
        cur_tail_idx -= shifts

        for b, (shift,batch_idx) in enumerate(zip(shifts, active_batches)):
            sl = (b, slice(cur_tokens.size(1)-shift,None)) # ...[b,-shift:]

            output_tokens[batch_idx].extend((
                [
                    decode(int(token)),
                    float(frame_times[token_idx]),
                ][typ]
                for typ, token, token_idx in zip(cur_types[sl], cur_tokens[sl], cur_token_idxs[sl])
            ))
            cur_types[sl] = 0
            cur_tokens[sl] = PAD
            cur_token_idxs[sl] = -1

        # remove completed samples from batch 
        next_active = cur_start_idxs < L
        if not next_active.any():
            # all completed
            break

        if not next_active.all():
            active_batches = active_batches[next_active]
            B = active_batches.size(0)
            cur_start_idxs = cur_start_idxs[next_active]
            cur_tail_idx = cur_tail_idx[next_active]
            cur_types = cur_types[next_active]
            cur_tokens = cur_tokens[next_active]
            cur_token_idxs = cur_token_idxs[next_active]
            c = c[next_active]

        # shrink sequence
        while cur_tokens.size(1) > 0 and (cur_tokens[:,-1] == PAD).all():
            cur_types = cur_types[:,:-1]
            cur_tokens = cur_tokens[:,:-1]
            cur_token_idxs = cur_token_idxs[:,:-1]
            
    pbar.close()

    for b, (tail_idx,batch_idx) in enumerate(zip(cur_tail_idx, active_batches)):
        sl = (b, slice(tail_idx)) # ...[b,:tail_idx]
        output_tokens[batch_idx].extend((
            [
                decode(int(token)),
                float(frame_times[token_idx]),
            ][typ]
            for typ, token, token_idx in zip(cur_types[sl], cur_tokens[sl], cur_token_idxs[sl])
        ))

    return output_tokens, th.stack([ th.stack(l).mean(dim=0) for l in pred_labels ])