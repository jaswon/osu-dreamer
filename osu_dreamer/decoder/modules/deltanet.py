
from jaxtyping import Float

import warnings
from dataclasses import dataclass

from torch import nn, Tensor

from fla.layers.gated_deltaproduct import GatedDeltaProduct

@dataclass
class DeltaNetArgs:
    head_dim: int
    n_heads: int
    n_householder: int

class DeltaNet(nn.Module):
    def __init__(
        self,
        dim: int,
        args: DeltaNetArgs,
    ):
        super().__init__()
        self.gdp = GatedDeltaProduct(
            hidden_size = dim,
            head_dim = args.head_dim,
            n_heads = args.n_heads,
            num_householder = args.n_householder,
            use_short_conv = False,
            allow_neg_eigval = True,
        )

    def forward(self, x: Float[Tensor, "B N D"]) -> Float[Tensor, "B N D"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, *_ = self.gdp(x)
            return o