
from jaxtyping import Float, Int

import torch as th
from torch.distributions import Kumaraswamy
from torch import nn, Tensor
import torch.nn.functional as F

import osu_dreamer.modules.mp as MP

class IntervalEmbedding(nn.Module):
    def __init__(self, interval_len: int, dim: int):
        super().__init__()
        self.proj = MP.Linear(1, dim)
        self.domain = th.linspace(0, 1, interval_len)

    def forward(self, x: Int[Tensor, "..."]) -> Float[Tensor, "... E"]:
        return self.proj(self.domain.to(x.device)[x,None])

class IntervalDist(nn.Module):
    def __init__(self, dim: int, interval_len: int):
        super().__init__()
        self.domain = th.linspace(0, 1, interval_len)
        self.interval = th.arange(interval_len)
        self.p = nn.Sequential(
            nn.Linear(dim, 2),
            nn.Softplus(),
        )

    def _dist(self, embs: Float[Tensor, "... E"]) -> th.distributions.Distribution:
        params = self.p(embs)+1
        return Kumaraswamy(params[...,[0]], params[...,[1]])

    def forward(
        self,
        embs: Float[Tensor, "... E"],
        target: Int[Tensor, "..."],
    ) -> Float[Tensor, ""]:
        """continuous ranked probability score"""
        D = embs.device
        points = self.domain.to(D).view(*([1] * (embs.ndim-1)), -1)
        pred_cdf = self._dist(embs).cdf(points)

        true_cdf = (self.interval.to(D) >= target[...,None]).float()
        return F.mse_loss(pred_cdf, true_cdf)
    
    def logits(
        self,
        embs: Float[Tensor, "... E"],
    ) -> Float[Tensor, "... N"]:
        D = embs.device
        points = self.domain.to(D).view(*([1] * (embs.ndim-1)), -1)
        return th.nan_to_num(self._dist(embs).log_prob(points), nan=-th.inf)