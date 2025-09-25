
from jaxtyping import Float, Int

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from osu_dreamer.lm.data.tokens.tokens import Vocab

from .sample import sample
    
class FourierFeatureEmbedding(nn.Module):
    def __init__(self, dim: int, n_freqs: int, sigma: float = 10.):
        super().__init__()
        self.ffe_freqs: th.Tensor
        self.register_buffer('ffe_freqs', 2 * th.pi * sigma * th.randn(dim, n_freqs))

    def forward(self, x: Float[Tensor, "*B D"]) -> Float[Tensor, "*B F"]:
        assert (x >= 0).all()
        assert (x <= 1).all()
        thetas = x @ self.ffe_freqs.to(x.device)  # [*B, n_freqs]
        return th.cat([th.sin(thetas), th.cos(thetas)], dim=-1)

class PositionHead(nn.Module):
    def __init__(
        self,
        vocab: Vocab,
        emb_dim: int,
        h_dim: int,
        n_freqs: int = 64,
        fake_to_real: int = 4,
    ):
        super().__init__()
        assert fake_to_real > 0
        self.fake_to_real = fake_to_real
        self.domain = vocab.x_min, vocab.x_max, vocab.y_min, vocab.y_max
        self.ffe = FourierFeatureEmbedding(2, n_freqs)
        self.emb_net = nn.Sequential(
            nn.Linear(2*n_freqs, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, emb_dim),
        )
        self.score_net = nn.Sequential(
            nn.Linear(2*n_freqs + emb_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, h_dim),
            nn.GELU(),
            nn.Linear(h_dim, 1),
        )

    def _coord_features(self, coord: Int[Tensor, "*B 2"]) -> Float[Tensor, "*B F"]:
        x_min, x_max, y_min, y_max = self.domain
        return self.ffe(th.stack([
            (coord[...,0] - x_min) / (x_max - x_min),
            (coord[...,1] - y_min) / (y_max - y_min),
        ], dim=-1))

    def embed(self, coord: Int[Tensor, "*B 2"]) -> Float[Tensor, "*B D"]:
        return self.emb_net(self._coord_features(coord))

    def _score(
        self, 
        emb: Float[Tensor, "*B D"], 
        coord: Int[Tensor, "*B 2"],
    ) -> Float[Tensor, "*B"]:
        features = th.cat([ emb, self._coord_features(coord) ], dim=-1)
        return self.score_net(features).squeeze(-1)
    
    def _sample_noise_coords(self, size: tuple[int, ...], d: th.device) -> Int[Tensor, "*B 2"]:
        x_min, x_max, y_min, y_max = self.domain
        return th.stack([
            th.randint(x_min, x_max, size, device=d),
            th.randint(y_min, y_max, size, device=d),
        ], dim=-1)
    
    @th.no_grad()
    def _log_probs_grid(
        self,
        embs: Float[Tensor, "B D"], 
        x_size: int, 
        y_size: int,
    ) -> tuple[
        Float[Tensor, "B N"],
        Int[Tensor, "N 2"],
    ]:
        x_min, x_max, y_min, y_max = self.domain

        assert (x_max - x_min) % x_size == 0
        assert (y_max - y_min) % y_size == 0

        grid = th.stack(th.meshgrid(
            th.arange(x_min, x_max, x_size) + x_size // 2,
            th.arange(y_min, y_max, y_size) + y_size // 2, 
            indexing='ij',
        ), dim=-1).view(-1, 2).to(embs.device) # N 2

        log_probs = F.log_softmax(self._score(
            embs[:,None,:].expand(-1, grid.size(0), -1),
            grid[None,:,:].expand(embs.size(0), -1, -1),
        ), dim=-1) # B N

        return log_probs, grid
    
    def forward(
        self,
        pred_embs: Float[Tensor, "B D"],
        true_positions: Int[Tensor, "B 2"],
        validation: bool = False,
    ) -> tuple[
        Float[Tensor, ""],  # loss
        dict[str, float],   # log dict
    ]:
        """noise contrastive estimation"""

        B = pred_embs.size(0)

        embs = pred_embs[:,None,:].expand(-1, 1+self.fake_to_real, -1)
        fake_positions = self._sample_noise_coords((B, self.fake_to_real), pred_embs.device) # B N 2
        positions = th.cat([true_positions[:,None,:], fake_positions], dim=1)

        scores = self._score(embs, positions) # B N+1
        labels = th.tensor([[1] + [0] * self.fake_to_real], device=pred_embs.device, dtype=th.float).expand(B, -1)

        loss = F.binary_cross_entropy_with_logits(scores, labels)
        log_dict = { "loss": loss.detach().item() }

        if validation:
            with th.no_grad():
                log_probs, grid = self._log_probs_grid(pred_embs, 32, 32)
                log_dict["val/entropy"] = -(log_probs.exp() * log_probs).sum(dim=1).mean().item()

                true_classes = (true_positions[:,None] - grid[None,:]).pow(2).sum(dim=-1).argmin(dim=1) # B
                log_dict["val/nll"] = -log_probs[th.arange(true_classes.size(0)), true_classes].mean().item()

                top_preds = log_probs.argsort(dim=1, descending=True)
                for k in [1, 5, 10]:
                    top_k_matches = (top_preds[:,:k] == true_classes[:,None]).any(dim=1)
                    log_dict[f"val/top_{k}"] = top_k_matches.float().mean().item()

        return loss, log_dict
    
    @th.no_grad()
    def sample(
        self, 
        embs: Float[Tensor, "*B D"], 
        x_size: int, 
        y_size: int,
        top_p: float,
    ) -> Int[Tensor, "*B 2"]:
        *b, d = embs.size()
        log_probs, grid = self._log_probs_grid(embs.view(-1, d), x_size, y_size)
        return grid[sample(log_probs.exp(), top_p)].view(*b, 2)