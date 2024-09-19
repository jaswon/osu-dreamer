
from dataclasses import dataclass
from jaxtyping import Float

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

@dataclass
class DropBlockArgs:
    block_size: int
    drop_prob: float
    sch_steps: int


class DropBlock1d(nn.Module):
    def __init__(
        self,
        args: DropBlockArgs,
    ):
        super().__init__()
        self.block_size = args.block_size
        self.drop_prob = args.drop_prob
        self.sch_steps = args.sch_steps
        self.register_buffer('step', th.tensor(0))

    def forward(
        self,
        x: Float[Tensor, "B D L"],
    ) -> Float[Tensor, "B D L"]:
        if not self.training or self.drop_prob == 0.0:
            return x
        
        self.step = th.clamp(self.step + 1, max=self.sch_steps)
        drop_prob = self.step.item() / self.sch_steps * self.drop_prob
        
        gamma = drop_prob / self.block_size
        mask = (th.rand_like(x) < gamma).float()
        mask = F.max_pool1d(mask, self.block_size, 1, self.block_size // 2)
        mask = 1 - mask
        return x * mask * mask.numel() / mask.sum()