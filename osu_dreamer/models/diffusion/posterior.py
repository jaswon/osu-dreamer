
from dataclasses import dataclass
from jaxtyping import Float

import torch as th
from torch import nn, Tensor

from .backbone import Backbone, BackboneArgs

@dataclass
class FlowPosteriorArgs:
    backbone_dim: int
    backbone_args: BackboneArgs

class FlowPosterior(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        flow_latent_dim: int,
        args: FlowPosteriorArgs,
    ):
        super().__init__()
        self.cls_token = nn.Parameter(th.randn(1, args.backbone_dim, 1))
        self.proj_in = nn.Conv1d(emb_dim, args.backbone_dim, 1)
        self.net = Backbone(args.backbone_dim, 0, 0, args.backbone_args)
        self.proj_out = nn.Linear(args.backbone_dim, flow_latent_dim)

    def forward(self, x: Float[Tensor, "B E l"]) -> Float[Tensor, "B Z"]:
        h = self.proj_in(x)
        cls = self.cls_token.expand(x.size(0),-1,1)
        h = self.net(th.cat([cls, h], dim=-1))[:,:,0] # B H
        return self.proj_out(h)