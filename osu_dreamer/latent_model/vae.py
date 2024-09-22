
from jaxtyping import Float

from dataclasses import dataclass

import torch as th
from torch import nn, Tensor

from osu_dreamer.data.beatmap.encode import X_DIM

from osu_dreamer.modules.wavenet import WaveNet, WaveNetArgs

class Residual(nn.Module):
    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return x + self.net(x)

@dataclass
class VAEArgs:
    beta: float

    latent_dim: int
    h_dim: int
    
    wavenet_args: WaveNetArgs

class VAE(nn.Module):
    def __init__(
        self,
        args: VAEArgs,
    ):
        super().__init__()
        self.beta = args.beta

        block = lambda dim: Residual(nn.Sequential(
            nn.Conv1d(dim, dim, 3,1,1, groups=dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),
        ))

        self.encoder = nn.Sequential(
            nn.Conv1d(X_DIM, args.h_dim, 1),
            WaveNet(args.h_dim, args.wavenet_args, block, transpose=False),
            nn.Conv1d(args.h_dim, args.latent_dim * 2, 1),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(args.latent_dim, args.h_dim, 1),
            WaveNet(args.h_dim, args.wavenet_args, block, transpose=True),
            nn.Conv1d(args.h_dim, X_DIM, 1),
        )

    def _encoder(
        self, 
        x: Float[Tensor, str(f"B {X_DIM} L")]
    ) -> tuple[
        Float[Tensor, "B Z L"], # mean
        Float[Tensor, "B Z L"], # logvar
    ]:
        return self.encoder(x).chunk(2, dim=1)

    def _reparam(
        self,
        mean: Float[Tensor, "B Z L"],
        logvar: Float[Tensor, "B Z L"],
    ) -> Float[Tensor, "B Z L"]:
        return mean + th.randn_like(mean) * th.exp(logvar * .5)
    
    def _decoder(
        self, 
        z: Float[Tensor, "B Z L"]
    ) -> Float[Tensor, str(f"B {X_DIM} L")]:
        return self.decoder(z)

    def forward(
        self, 
        x: Float[Tensor, str(f"B {X_DIM} L")]
    ) -> tuple[Float[Tensor, ""], dict[str, Float[Tensor, ""]]]:
        mean, logvar = self._encoder(x)
        z = self._reparam(mean, logvar)
        x_hat = self._decoder(z)

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

    def encode(
        self, 
        x: Float[Tensor, str(f"B {X_DIM} L")]
    ) -> Float[Tensor, "B Z L"]:
        return self._reparam(*self._encoder(x))

    def decode(
        self, 
        z: Float[Tensor, "B Z L"]
    ) -> Float[Tensor, str(f"B {X_DIM} L")]:
        return self._decoder(z).clamp(min=-1, max=1)