
from dataclasses import dataclass
from jaxtyping import Float

from torch import nn, Tensor

@dataclass
class LabelPredictorArgs:
    h_dim: int

class LabelPredictor(nn.Module):
    def __init__(self, x_dim: int, label_dim: int, args: LabelPredictorArgs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(x_dim, args.h_dim),
            nn.SiLU(),
            nn.Linear(args.h_dim, label_dim),
        )

    def forward(self, x: Float[Tensor, "B X"]) -> Float[Tensor, "B C"]:
        return 5 + 5 * self.net(x)