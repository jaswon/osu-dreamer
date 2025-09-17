
from jaxtyping import Float, Int

import torch as th
import torch.nn.functional as F
from torch import nn, Tensor

from osu_dreamer.lm.data.tokens.tokens import Vocab
    
class CategoricalHead(nn.Module):
    """
    Standard classification head for discrete categorical variables.
    
    Designed for cases where:
    - Classes represent distinct categories with no natural ordering
    - Each class should be treated as equally different from others
    """

    def __init__(self, emb_dim: int, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.emb = nn.Embedding(n_classes, emb_dim)
        self.head = nn.Linear(emb_dim, n_classes)

    def embed(self, i: Int[Tensor, "*B"]) -> Float[Tensor, "*B D"]:
        assert (i>=0).all()
        assert (i<self.n_classes).all()
        return self.emb(i)
    
    def logits(self, embs: Float[Tensor, "*B D"]) -> Float[Tensor, "*B C"]:
        return self.head(embs)

    def forward(
        self,
        pred_embs: Float[Tensor, "*B D"],
        true_classes: Int[Tensor, "*B"],
        validation: bool = False,
    ) -> tuple[
        Float[Tensor, ""],  # loss
        dict[str, float],   # log dict
    ]:
        pred_logits = self.logits(pred_embs)
        loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)), 
            true_classes.reshape(-1),
        )
        return loss, { "loss": loss.detach().item() }
    
class FourierFeatureEmbedding(nn.Module):
    def __init__(self, dim: int, emb_dim: int, n_freqs: int = 64, sigma: float = 10.):
        super().__init__()
        self.ffe_freqs: th.Tensor
        self.register_buffer('ffe_freqs', 2 * th.pi * sigma * th.randn(dim, n_freqs))
        self.ffe_proj = nn.Linear(2 * n_freqs, emb_dim)

    def forward(self, x: Float[Tensor, "*B D"]) -> Float[Tensor, "*B E"]:
        assert (x >= 0).all()
        assert (x < 1).all()
        thetas = x @ self.ffe_freqs  # [*B, n_freqs]
        features = th.cat([th.sin(thetas), th.cos(thetas)], dim=-1)
        return self.ffe_proj(features)

class SpatialCategory1dHead(nn.Module):
    """
    Classification head for ordered categorical variables representing discretized continuous values.
    
    Designed for cases where:
    - Classes represent bins of an underlying continuous 1D space (e.g., time, position, angle)
    - Neighboring classes should be treated as more similar than distant ones
    """

    def __init__(self, emb_dim: int, n_classes: int):
        super().__init__()
        self.n_classes = n_classes
        self.emb = FourierFeatureEmbedding(1, emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, n_classes-1),
        ) # V[k] = log(P[k+1] / P[k]) (adjacent-category logit)

    def embed(self, i: Int[Tensor, "*B"]) -> Float[Tensor, "*B D"]:
        assert (i>=0).all()
        assert (i<self.n_classes).all()
        return self.emb(i[:,:,None] / self.n_classes)
    
    def logits(self, embs: Float[Tensor, "*B D"]) -> Float[Tensor, "*B C"]:
        # P(i+1) = ( P(i+1) / P(i) ) * P(i)
        return F.pad(self.head(embs).cumsum(dim=-1), (1,0))

    def forward(
        self,
        pred_embs: Float[Tensor, "*B D"],
        true_classes: Int[Tensor, "*B"],
        validation: bool = False,
    ) -> tuple[
        Float[Tensor, ""],  # loss
        dict[str, float],   # log dict
    ]:
        logits = self.logits(pred_embs)
        loss = F.cross_entropy(logits, true_classes)
        log_dict = { "loss": loss.detach().item() }

        if validation:
            with th.no_grad():
                # top-1 class mean absolute error
                pred_classes = logits.argmax(dim=-1)
                log_dict["mae"] = (pred_classes - true_classes).abs().float().mean().item()

                # entropy (higher = more uncertain)
                probs = logits.softmax(dim=-1)
                entropy = -(probs * th.log(probs + 1e-8)).sum(dim=-1)
                log_dict["entropy"] = entropy.mean().item()

        return loss, log_dict

class TokenHead(nn.Module):
    def __init__(
        self,
        vocab: Vocab,
        emb_dim: int,
        time_loss_factor: float = 1.,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.time_loss_factor = time_loss_factor
        self.vocab = vocab

        num_token_classes = vocab.T0 + 1 # token classes + one time sentinel
        num_time_classes = vocab.time_bins+1 # time classes + EOS

        self.time_head = SpatialCategory1dHead(emb_dim, num_time_classes)
        self.token_head = CategoricalHead(emb_dim, num_token_classes)

    def to_head_ids(self, ids: Int[Tensor, "B N"]) -> tuple[
        Int[Tensor, "B N"], # modes
        Int[Tensor, "B N"], # token ids
        Int[Tensor, "B N"], # time ids
    ]:
        assert (ids >= 0).all()
        assert (ids < len(self.vocab.tokens)).all()

        modes = (ids >= self.vocab.T0).long()
        imodes = modes[None].expand(2, -1, -1) # used for gather

        t0 = th.full_like(ids, self.vocab.T0)

        token_ids = th.stack([
            ids,
            t0, # mode=1 (time) - collapse to T0
        ]).gather(dim=0, index=imodes)[0]

        time_ids = th.stack([
            t0, # mode=0 (token) - collapse to T0 (ignored by mask)
            ids - self.vocab.T0,
        ]).gather(dim=0, index=imodes)[0]

        return modes, token_ids, time_ids

    def embed(self, ids: Int[Tensor, "B N"]) -> Float[Tensor, "B N D"]:
        modes, i, t = self.to_head_ids(ids)
        embs = th.stack([
            self.token_head.embed(i),
            self.time_head.embed(t),
        ]) # K B N D
        return embs.gather(dim=0, index=modes[None,:,:,None].expand_as(embs))[0]
    
    def log_probs(self, pred_embs: Float[Tensor, "B N D"]) -> Float[Tensor, "B N V"]:
        token_log_probs = F.log_softmax(self.token_head.logits(pred_embs), dim=-1)
        time_log_probs = F.log_softmax(self.time_head.logits(pred_embs), dim=-1)
        
        # P(time_i) = P(time_i | time_type) * P(time_type)
        marginal_time_log_probs = token_log_probs[:,:,-1:] + time_log_probs
        return th.cat([
            token_log_probs[:,:,:-1],   # non-time tokens
            marginal_time_log_probs,    # time tokens
        ], dim=-1)

    def forward(
        self,
        pred_embs: Float[Tensor, "B N D"],
        true_classes: Int[Tensor, "B N"],
        validation: bool = False,
    ) -> tuple[
        Float[Tensor, ""],  # loss
        dict[str, float],   # log dict
    ]:
        modes, true_token_ids, true_time_ids = self.to_head_ids(true_classes)

        token_loss, token_loss_dict = self.token_head(
            pred_embs, 
            true_token_ids, 
            validation=validation,
        )

        is_time = modes == 1
        if is_time.any():
            time_loss, time_loss_dict = self.time_head(
                pred_embs[is_time], 
                true_time_ids[is_time], 
                validation=validation,
            )
        else:
            time_loss, time_loss_dict = th.tensor(0, device=pred_embs.device), {}

        loss = token_loss + self.time_loss_factor * time_loss
        log_dict = {
            **{ f"token/{k}":v for k,v in token_loss_dict.items() },
            **{ f"time/{k}":v for k,v in time_loss_dict.items() },
            "loss/total": loss.detach().item(),
        }

        if validation:
            with th.no_grad():
                pred_log_probs = self.log_probs(pred_embs)
                is_token = modes == 0

                # Top-k accuracy
                for k in [1, 5, 10]:
                    top_k_preds = pred_log_probs.topk(k, dim=-1)[1]  # B N k
                    top_k_matches = (top_k_preds == true_classes.unsqueeze(-1)).any(dim=-1)

                    log_dict[f"acc/total/top_{k}"] = top_k_matches.float().mean().item()

                    if is_token.any():
                        log_dict[f"acc/token/top_{k}"] = top_k_matches[is_token].float().mean().item()
                    
                    if is_time.any():
                        log_dict[f"acc/time/top_{k}"] = top_k_matches[is_time].float().mean().item()
                
        return loss, log_dict