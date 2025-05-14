
from jaxtyping import Float, Int

import torch.nn.functional as F
from torch import Tensor


def focal_loss(
    inputs: Float[Tensor, "N D ..."],
    target: Int[Tensor, "N ..."],
    gamma: float,
    weight: None | Float[Tensor, "D"] = None,
) -> Float[Tensor, "N ..."]:
    logpt = F.log_softmax(inputs, dim=1)
    inputs = (1 - logpt.exp()).pow(gamma) * logpt
    return F.nll_loss(inputs, target, weight, reduction='none')