
from jaxtyping import Float, Int

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from osu_dreamer.lm.data.tokens.tokens import Vocab
    
class FourierFeatureEmbedding(nn.Module):
    def __init__(self, dim: int, n_freqs: int, sigma: float = 100.):
        super().__init__()
        self.ffe_freqs: th.Tensor
        self.register_buffer('ffe_freqs', 2 * th.pi * sigma * th.randn(dim, n_freqs))

    def forward(self, x: Float[Tensor, "*B D"]) -> Float[Tensor, "*B F"]:
        assert (x >= 0).all()
        assert (x < 1).all()
        thetas = x @ self.ffe_freqs.to(x.device)  # [*B, n_freqs]
        return th.cat([th.sin(thetas), th.cos(thetas)], dim=-1)

class PositionHead(nn.Module):
    def __init__(
        self,
        vocab: Vocab,
        emb_dim: int,
        h_dim: int,
        n_freqs: int = 128,
    ):
        super().__init__()
        self.domain = vocab.x_min, vocab.x_max, vocab.y_min, vocab.y_max
        self.emb_net = nn.Sequential(
            FourierFeatureEmbedding(2, n_freqs),
            nn.Linear(2*n_freqs, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, emb_dim),
        )
        self.head_net = nn.Sequential(
            nn.Linear(emb_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 4),
        )

    def _norm_coords(self, coords: Int[Tensor, "*B 2"]) -> Float[Tensor, "*B 2"]:
        x_min, x_max, y_min, y_max = self.domain
        return th.stack([
            (coords[...,0] - x_min) / (x_max - x_min),
            (coords[...,1] - y_min) / (y_max - y_min),
        ], dim=-1)
    
    def _denorm_coords(self, norms: Float[Tensor, "*B 2"]) -> Int[Tensor, "*B 2"]:
        x_min, x_max, y_min, y_max = self.domain
        return th.stack([
            x_min + (x_max - x_min) * norms[...,0],
            y_min + (y_max - y_min) * norms[...,1],
        ], dim=-1).long()

    def embed(self, coords: Int[Tensor, "*B 2"]) -> Float[Tensor, "*B D"]:
        return self.emb_net(self._norm_coords(coords))
    
    def get_dists(self, embs: Float[Tensor, "*B D"]) -> tuple[th.distributions.Beta, th.distributions.Beta]:
        params = 1 + F.softplus(self.head_net(embs)) # alpha, beta > 1
        x_alpha, x_beta, y_alpha, y_beta = params.unbind(dim=-1)
        return (
            th.distributions.Beta(x_alpha, x_beta),
            th.distributions.Beta(y_alpha, y_beta),
        )
    
    def forward(
        self,
        pred_embs: Float[Tensor, "B D"],
        true_positions: Int[Tensor, "B 2"],
        validation: bool = False,
    ) -> tuple[
        Float[Tensor, ""],  # loss
        dict[str, float],   # log dict
    ]:
        X, Y = self.get_dists(pred_embs)
        coords = self._norm_coords(true_positions)
        nll = -( X.log_prob(coords[:,0]) + Y.log_prob(coords[:,1]) ).mean()

        loss = nll
        log_dict = {
            "loss": loss.detach().item(),
            "nll": nll.detach().item(),
        }

        if validation:
            with th.no_grad():
                log_dict["val/entropy"] = (X.entropy() + Y.entropy()).mean().item()

        return loss, log_dict
    
    @th.no_grad()
    def sample(
        self, 
        embs: Float[Tensor, "*B D"],
        top_p: float,
    ) -> Int[Tensor, "*B 2"]:
        X, Y = self.get_dists(embs)
        return self._denorm_coords(th.stack([ X.sample(), Y.sample() ], dim=-1))