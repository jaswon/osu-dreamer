
from jaxtyping import Float, Int

import math

import torch as th
import torch.nn.functional as F
from torch import nn, Tensor

from osu_dreamer.lm.data.tokens.tokens import Vocab
    

class TokenHead(nn.Module):
    def __init__(
        self,
        vocab: Vocab,
        emb_dim: int,
        n_freqs: int = 64,
        time_loss_factor: float = 1.,
    ):
        super().__init__()
        self.time_loss_factor = time_loss_factor
        self.vocab = vocab
        num_tokens = vocab.T0 + 1 # arbitrary for non-tokens + one time sentinel token

        ## embedding
        self.token_emb = nn.Embedding(num_tokens, emb_dim)

        # RFF for time embeddings
        rff_freqs = th.exp(th.randn(n_freqs) * math.log(vocab.time_bins))
        self.register_buffer("rff_freqs", rff_freqs)
        self.rff_freqs: Tensor
        self.rff_proj = nn.Linear(2 * n_freqs, emb_dim)

        ## prediction head
        self.token_head = nn.Linear(emb_dim, num_tokens)
        self.time_head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, vocab.time_bins-1),
        ) # V[k] = log(P[k+1] / P[k]) (adjacent-category logit)

    def embed(self, ids: Int[Tensor, "B N"]) -> Float[Tensor, "B N D"]:
        positions = th.clamp(ids - self.vocab.T0, min=0) # B N
        thetas = positions[:,:,None] * self.rff_freqs[None,None]  # [B, N, n_freqs]
        features = th.cat([th.sin(thetas), th.cos(thetas)], dim=-1)
        time_embs = self.rff_proj(features) # B N D

        return th.where(
            (ids >= self.vocab.T0)[:,:,None],
            time_embs,
            self.token_emb(th.clamp(ids, max=self.vocab.T0)),
        )
    
    def logits(self, pred_embs: Float[Tensor, "B N D"]) -> Float[Tensor, "B N V"]:
        return self._denormalized_logits(
            self.token_head(pred_embs),
            self.time_head(pred_embs),
        )
        
    def _denormalized_logits(
        self,
        token_scores: Float[Tensor, "B N D"],
        time_scores: Float[Tensor, "B N Tm1"],
    ) -> Float[Tensor, "B N V"]:
        
        # P(time_i+1) = ( P(time_i+1) / P(time_i) ) * P(time_i)
        time_logits = F.pad(time_scores.cumsum(dim=-1), (1,0))

        token_probs = token_scores.softmax(dim=-1)
        time_probs = time_logits.softmax(dim=-1)
        
        # P(time_i) = P(time_i | time_type) * P(time_type)
        marginal_time_probs = token_probs[:,:,-1:] * time_probs
        combined_probs = th.cat([token_probs[:,:,:-1], marginal_time_probs], dim=-1)
        return th.log(combined_probs + 1e-8)

    def forward(
        self,
        pred_embs: Float[Tensor, "B N D"],
        true_tokens: Int[Tensor, "B N"],
        validation: bool = False,
    ) -> tuple[
        Float[Tensor, ""],  # loss
        dict[str, float],   # log dict
    ]:
        is_time = true_tokens >= self.vocab.T0

        pred_token_scores = self.token_head(pred_embs)
        true_token_classes = th.clamp(true_tokens, max=self.vocab.T0)
        token_loss = F.cross_entropy(
            pred_token_scores.reshape(-1, pred_token_scores.size(-1)), 
            true_token_classes.reshape(-1),
        )

        # adjacent-category ordinal regression
        pred_time_scores = self.time_head(pred_embs)
        pred_time_logits = F.pad(pred_time_scores.cumsum(dim=-1), (1,0))
        true_time_classes = th.clamp(true_tokens - self.vocab.T0, min=0)
        time_loss = F.cross_entropy(
            pred_time_logits[is_time],
            true_time_classes[is_time],
        )

        loss = token_loss + self.time_loss_factor * time_loss
        log_dict = {
            "loss/token": token_loss.detach().item(),
            "loss/time": time_loss.detach().item(),
            "loss/total": loss.detach().item(),
        }

        if validation:
            with th.no_grad():
                pred_logits = self._denormalized_logits(pred_token_scores, pred_time_scores)

                if is_time.any():
                    # Time class MAE (Mean Absolute Error)
                    pred_tokens = pred_logits.argmax(dim=-1)
                    pred_time_classes = th.clamp(pred_tokens[is_time] - self.vocab.T0, min=0)
                    true_time_classes_val = th.clamp(true_tokens[is_time] - self.vocab.T0, min=0)
                    time_mae = (pred_time_classes - true_time_classes_val).abs().float().mean()
                    log_dict["time_mae"] = time_mae.item()

                    # Entropy of time predictions (higher = more uncertain)
                    time_probs = pred_time_logits.softmax(dim=-1)
                    time_entropy = -(time_probs * th.log(time_probs + 1e-8)).sum(dim=-1)
                    avg_time_entropy = time_entropy[is_time].mean()
                    log_dict["time_entropy"] = avg_time_entropy.item()
                    
                    # Timing bias: mean predicted time vs true time
                    pred_time_mean = pred_time_classes.float().mean()
                    true_time_mean = true_time_classes_val.float().mean()
                    time_bias = pred_time_mean - true_time_mean
                    log_dict["time_bias"] = time_bias.item()
                
                is_non_time = ~is_time

                # Top-k accuracy
                for k in [1, 5, 10]:
                    top_k_preds = pred_logits.topk(k, dim=-1)[1]  # B N k
                    top_k_matches = (top_k_preds == true_tokens.unsqueeze(-1)).any(dim=-1)
                    top_k_accuracy = top_k_matches.float().mean()
                    log_dict[f"acc/total/top_{k}"] = top_k_accuracy.item()

                    # Token-specific accuracy (non-time tokens)
                    if is_non_time.any():
                        token_accuracy = top_k_matches[is_non_time].float().mean()
                        log_dict[f"acc/token/top_{k}"] = token_accuracy.item()
                    
                    # Top-k accuracy for time tokens specifically
                    if is_time.any():
                        time_accuracy = top_k_matches[is_time].float().mean()
                        log_dict[f"acc/time/top_{k}"] = time_accuracy.item()
                
        return loss, log_dict