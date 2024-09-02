
from dataclasses import dataclass

from jaxtyping import Float

import torch as th
import torch.nn.functional as F
from torch import nn, Tensor

from einops import repeat

from osu_dreamer.data.prepare_map import NUM_LABELS

@dataclass
class LabelGenArgs:
    gf_feats: int
    h_dim: int
    gf_scale: float = 30.
    beta: float = 1.

class LabelGen(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        args: LabelGenArgs,
    ):
        super().__init__()
        self.beta = args.beta

        self.gf_feats = args.gf_feats
        self.gfW = nn.Parameter(th.randn(args.gf_feats * NUM_LABELS) * args.gf_scale, requires_grad=False)

        self.encoder = nn.Sequential(
            nn.Linear(args.gf_feats * NUM_LABELS * 2, args.h_dim),
            nn.SiLU(),
            nn.Linear(args.h_dim, emb_dim * 2),
        )

        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, args.h_dim),
            nn.SiLU(),
            nn.Linear(args.h_dim, NUM_LABELS),
        )

    def encode(
        self,
        x: Float[Tensor, str(f"B {NUM_LABELS}")],
    ) -> tuple[Float[Tensor, "B Z"], Float[Tensor, "B Z"]]:
        theta = repeat(x, 'b l -> b (l f)', f = self.gf_feats) * self.gfW * 2 * th.pi
        gf = th.cat([theta.sin(), theta.cos()], dim=-1)
        return self.encoder(gf).chunk(2, dim=1)
    
    def reparametrize(
        self,
        mu: Float[Tensor, "B Z"], 
        logvar: Float[Tensor, "B Z"],
    ) -> Float[Tensor, "B Z"]:
        return mu + th.exp(logvar) * th.randn_like(mu)
    
    def embedding(
        self,
        x: Float[Tensor, str(f"B {NUM_LABELS}")],
    ) -> Float[Tensor, "B Z"]:
        return self.reparametrize(*self.encode(x))

    def forward(
        self,
        x: Float[Tensor, str(f"B {NUM_LABELS}")],
    ) -> tuple[
        Float[Tensor, "B Z"],
        Float[Tensor, ""],
    ]:
        x_masked = x.masked_fill(th.rand_like(x) > .5, 0.)
        mu, logvar = self.encode(x_masked)
        z = self.reparametrize(mu, logvar)

        x_hat = self.decoder(z)

        loss_recon = F.mse_loss(x_hat, x)
        loss_kl = .5 * ( th.exp(2 * logvar) + mu ** 2 - logvar - 1 ).mean()

        loss = loss_recon + self.beta * loss_kl

        return z, loss