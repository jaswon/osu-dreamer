
from jaxtyping import Float, Int

import torch as th

def sample(
    probs: Float[th.Tensor, "*B N"], 
    top_p: float,
) -> Int[th.Tensor, "*B 1"]:

    if top_p <= 0:
        # Greedy sampling
        return th.argmax(probs, dim=-1, keepdim=True)
    else:
        # Nucleus sampling
        sorted_probs, sorted_indices = th.sort(probs, descending=True)
        cutoff_mask = sorted_probs.cumsum(dim=-1) > top_p
        cutoff_mask[..., 0] = False
        masked_logits = sorted_probs.masked_fill(cutoff_mask, 0.0).log()
        return sorted_indices.gather(-1, masked_logits.softmax(dim=-1).multinomial(num_samples=1))