
from dataclasses import dataclass
from jaxtyping import Float

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
        self.proj_in = nn.Conv1d(emb_dim, args.backbone_dim, 1)
        self.net = Backbone(args.backbone_dim, 0, args.backbone_args)
        self.proj_out = nn.Linear(args.backbone_dim, flow_latent_dim)

    def forward(self, x: Float[Tensor, "B E l"]) -> Float[Tensor, "B Z"]:
        h = self.proj_in(x)
        h = self.net(h).mean(dim=-1) # B H
        return self.proj_out(h)