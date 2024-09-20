
from jaxtyping import Float

from dataclasses import dataclass

import torch as th
from torch import nn, Tensor

from osu_dreamer.data.beatmap.encode import X_DIM

from osu_dreamer.modules.unet import UNet

@dataclass
class VAEArgs:
    beta: float

    latent_dim: int
    h_dim: int
    
    scales: list[int]

class VAE(nn.Module):
    def __init__(
        self,
        args: VAEArgs,
    ):
        super().__init__()
        self.beta = args.beta

        block = lambda: nn.Sequential(
            nn.Conv1d(args.h_dim, args.h_dim, 5,1,2, groups=args.h_dim),
            nn.Conv1d(args.h_dim, args.h_dim, 1),
            nn.SiLU(),
            nn.Conv1d(args.h_dim, args.h_dim, 1),
        )

        self.encoder = nn.Sequential(
            nn.Conv1d(X_DIM, args.h_dim, 1),
            UNet(args.h_dim, args.scales, block(), block),
            nn.Conv1d(args.h_dim, args.latent_dim * 2, 1),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(args.latent_dim, args.h_dim, 1),
            UNet(args.h_dim, args.scales, block(), block),
            nn.Conv1d(args.h_dim, X_DIM, 1),
        )

    def reparam(
        self,
        mean: Float[Tensor, "B Z L"],
        logvar: Float[Tensor, "B Z L"],
    ) -> Float[Tensor, "B Z L"]:
        return mean + th.randn_like(mean) * th.exp(logvar * .5)

    def encode(self, x: Float[Tensor, str(f"B {X_DIM} L")]) -> Float[Tensor, "B Z L"]:
        return self.reparam(*self.encoder(x).chunk(2, dim=1))

    def decode(self, z: Float[Tensor, "B Z L"]) -> Float[Tensor, str(f"B {X_DIM} L")]:
        return self.decoder(z).clamp(min=-1, max=1)

    def forward(self, x: Float[Tensor, str(f"B {X_DIM} L")]) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        mean, logvar = self.encoder(x).chunk(2, dim=1)
        z = self.reparam(mean, logvar)
        x_hat = self.decoder(z)

        recon_loss = th.mean((x - x_hat) ** 2)
        bound_loss = th.mean((x_hat.abs().clamp(min=1) - 1) ** 2)
        kl_loss = .5 * (mean ** 2 + logvar.exp() - logvar - 1).sum(dim=1).mean()

        loss = recon_loss + bound_loss + self.beta * kl_loss

        return loss, {
            'loss': loss.detach(),
            'recon': recon_loss.detach(),
            'bound': bound_loss.detach(),
            'kl': kl_loss.detach(),
        }