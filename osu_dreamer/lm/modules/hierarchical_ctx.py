
from jaxtyping import Float, Int

import torch as th
from torch import nn, Tensor

class HierarchicalEncoder(nn.Module):

    class Encoded:
        def __init__(
            self,
            r: list[tuple[int, int]],
            scales: list[Float[Tensor, "B D L"]],
        ):
            self.r = r
            self.scales = scales

        def slice(self, positions: Int[Tensor, "B N"]) -> Float[Tensor, "B N T D"]:
            ctx = []
            last_w = 1
            b = th.arange(positions.size(0))[:,None]
            for s, ((r_past, r_future), scale) in enumerate(zip(self.r, self.scales)):
                scale = scale.transpose(1,2) # B L D
                for i in range(-r_past, r_future+1):
                    if s==0 or i!=0:
                        ctx.append(scale[b, positions+i*last_w]) # B N D
                last_w *= 1+r_past+r_future
            return th.stack(ctx, dim=2)

    def __init__(
        self,
        d: int,
        r: list[tuple[int, int]],
    ):
        super().__init__()
        self.r = r

        last_w = 1
        self.convs = nn.ModuleList()
        for r_past, r_future in r[:-1]:
            w = 1+r_past+r_future
            self.convs.append(nn.Sequential(
                nn.ZeroPad1d((r_past*last_w, r_future*last_w)),
                nn.Conv1d(d, d, w, 1, dilation=last_w, groups=d),
                nn.Conv1d(d, d, 1),
            ))
            last_w *= w

    def forward(self, x: Float[Tensor, "B D L"]) -> Encoded:
        scales = [x]
        for conv in self.convs:
            scales.append(conv(scales[-1]))
        return self.Encoded(self.r, scales)