
from jaxtyping import Float

import warnings
from dataclasses import dataclass

from torch import nn, Tensor

from einops.layers.torch import Rearrange

from fla.ops.delta_rule import chunk_delta_rule

@dataclass
class DeltaNetArgs:
    head_dim: int
    n_heads: int
    value_dim: int = -1

class DeltaNet(nn.Module):
    def __init__(
        self,
        dim: int,
        args: DeltaNetArgs,
    ):
        super().__init__()
        self.value_dim = args.value_dim if args.value_dim > 0 else args.head_dim
        self.head_dim = args.head_dim
        h_dim = args.head_dim * args.n_heads
        self.proj_q = nn.Sequential(
            nn.Linear(dim, h_dim, bias=False),
            nn.SiLU(),
            Rearrange('... (h d) -> ... h d', d=args.head_dim),
        )
        self.proj_k = nn.Sequential(
            nn.Linear(dim, h_dim, bias=False),
            nn.SiLU(),
            Rearrange('... (h d) -> ... h d', d=args.head_dim),
        )
        h_v_dim = self.value_dim * args.n_heads
        self.proj_v = nn.Sequential(
            nn.Linear(dim, h_v_dim, bias=False),
            nn.SiLU(),
            Rearrange('... (h d) -> ... h d', d=self.value_dim),
        )
        self.proj_b = nn.Linear(dim, args.n_heads, bias=False)

        self.proj_o = nn.Sequential(
            nn.RMSNorm(args.head_dim),
            Rearrange('... h d -> ... (h d)'),
            nn.Linear(h_v_dim, dim, bias=False),
        )

        self.apply(self._initialize_weights)

    def _initialize_weights(self, module: nn.Module):
        if getattr(module, "_is_hf_initialized", False):
            return
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=2 ** -2.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        setattr(module, "_is_hf_initialized", True)

    def forward(self, x: Float[Tensor, "B N D"]) -> Float[Tensor, "B N D"]:

        beta = self.proj_b(x).sigmoid() * 2
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, _ = chunk_delta_rule(
                q,k,v,beta,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
        
        return self.proj_o(o)