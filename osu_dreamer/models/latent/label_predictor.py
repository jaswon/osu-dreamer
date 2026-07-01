
from dataclasses import dataclass
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

@dataclass
class LabelPredictorArgs:
    h_dim: int
    grad_factor: float

class LabelPredictor(nn.Module):
    def __init__(self, x_dim: int, label_dim: int, args: LabelPredictorArgs):
        super().__init__()
        self.grad_factor = args.grad_factor
        self.net = nn.Sequential(
            nn.Conv1d(x_dim, args.h_dim, 3),
            nn.SiLU(),
            nn.Conv1d(args.h_dim, label_dim, 1),
        )

    def forward(self, x: Float[Tensor, "B X l"]) -> Float[Tensor, "B C"]:
        if self.training:
            x = th.lerp(x.detach(), x, self.grad_factor)
        return 5 + 5 * self.net(x).mean(dim=2)