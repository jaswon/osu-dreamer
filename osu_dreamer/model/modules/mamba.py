
from jaxtyping import Float

from torch import nn, Tensor
import torch.nn.functional as F

from .s4 import S4Block as _S4Block

class S4Block(_S4Block):
    def __init__(self, dim: int):
        super().__init__(
            dim, 
            mode='diag', 
            bidirectional=True, 
            final_act=None, # type: ignore
            activation=None,
        )

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L"]:
        return super().forward(x.float())[0].type_as(x)

        ## chunked generation
        # state = self.seq.default_state(x.size(0))
        # ys = []
        # for u_ in h.float().chunk(4, dim=-1):
        #     y_, state = self.ssm(u_, state=state)
        #     ys.append(y_)
        # return th.cat(ys, dim=-1).type_as(x)

    
class MambaBlock(nn.Module):
    def __init__(self, dim: int, expand: int = 1):
        super().__init__()

        h_dim = dim * expand if expand > 0 else dim

        self.proj_in = nn.Conv1d(dim, h_dim*2, 1)
        self.seq = S4Block(h_dim)
        self.proj_out = nn.Sequential(
            nn.GroupNorm(1, h_dim),
            nn.Conv1d(h_dim, dim, 1),
        )

    def forward(self, x: Float[Tensor, "B D L"]) -> Float[Tensor, "B D L"]:
        gate, h = self.proj_in(x).chunk(2, dim=1)
        return self.proj_out(F.silu(gate) * self.seq(h))