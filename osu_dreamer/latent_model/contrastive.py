from jaxtyping import Float

import torch as th
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, reduce


def batch_contrastive_loss(
    z: Float[Tensor, "B D l"],
    temperature: float = 0.1,
) -> Float[Tensor, ""]:
    """
    Prototype-based contrastive (InfoNCE) loss treating each latent's source
    sample (batch index) as its class.

    Each sample is collapsed to a prototype (its mean latent). Every latent
    position is then an anchor whose positive is its own sample's prototype and
    whose negatives are every other sample's prototype. Minimizing this pulls
    same-sample latents toward their shared prototype and pushes them away from
    other samples' prototypes.
    """
    B, _, l = z.size()

    # L2-normalize so similarities are cosine
    feats = F.normalize(rearrange(z, 'b d l -> (b l) d'), dim=1)  # (N, D)

    # per-sample prototype = normalized mean latent
    protos = F.normalize(reduce(z, 'b d l -> b d', 'mean'), dim=1)  # (B, D)

    logits = feats @ protos.T / temperature  # (N, B)

    # positive prototype index for each anchor is its source sample
    target = th.arange(B, device=z.device).repeat_interleave(l)  # (N,)

    return F.cross_entropy(logits, target)

