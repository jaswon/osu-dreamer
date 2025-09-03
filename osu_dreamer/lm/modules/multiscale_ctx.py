
from jaxtyping import Float, Int

import torch as th
from torch import nn, Tensor

class MultiScaleEncoder(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, r: list[tuple[int, int]]):
        super().__init__()
        self.r = r
        self.out = nn.Linear(in_dim, out_dim)

        last_w = 1
        self.convs = nn.ModuleList()
        for r_past, r_future in r[:-1]:
            w = 1+r_past+r_future
            self.convs.append(nn.Sequential(
                nn.ZeroPad1d((r_past*last_w, r_future*last_w)),
                nn.Conv1d(in_dim, in_dim, w, 1, dilation=last_w, groups=in_dim),
                nn.Conv1d(in_dim, in_dim, 1),
            ))
            last_w *= w

    def precompute(self, x: Float[Tensor, "1 D L"]) -> list[Float[Tensor, "1 D L"]]:
        features = [x,x]
        for conv in self.convs:
            features.append(conv(features[-1]))
        return features

    def forward(
        self,
        features: list[Float[Tensor, "1 iD L"]],
        positions: Int[Tensor, "B N"],
    ) -> Float[Tensor, "B N T oD"]:
        assert len(features) == len(self.r) + 1

        L = features[0].size(-1)
        ctx = []
        stride = 1
        for x_conv, r in zip(features, [None] + self.r):
            p = positions[:, :, None]
            if r is not None:
                r_past, r_future = r
                w = th.cat([-1 - th.arange(r_past), 1 + th.arange(r_future)], dim=0)
                p = p + stride * w.to(positions.device)[None, None, :]  # B N W
                stride *= 1 + r_past + r_future
            ctx_p = x_conv.transpose(1, 2).contiguous()[0, th.clamp(p, 0, L - 1)]
            ctx_p[(p < 0) | (p >= L)] = 0
            ctx.append(ctx_p)  # B N W D

        return self.out(th.cat(ctx, dim=2))