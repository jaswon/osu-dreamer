
from jaxtyping import Float, Int

import math

import torch as th
import torch.nn.functional as F
from torch import nn, Tensor

from osu_dreamer.lm.data.tokens.tokens import Token, TokenType, Vocab
    

class TokenHead(nn.Module):
    def __init__(
        self,
        vocab: Vocab,
        emb_dim: int,
        n_freqs: int = 64,
        time_loss_factor: float = 100.,
    ):
        super().__init__()
        self.time_loss_factor = time_loss_factor
        self.vocab_size = len(vocab.tokens)
        self.t0 = vocab.ids[Token(TokenType.TIME, 0)]
        num_tokens = self.t0 + 1 # arbitrary for non-tokens + one time sentinel token

        ## embedding
        self.token_emb = nn.Embedding(num_tokens, emb_dim)

        # RFF for time embeddings
        rff_freqs = th.exp(th.randn(n_freqs) * math.log(vocab.time_bins))
        self.register_buffer("rff_freqs", rff_freqs)
        self.rff_freqs: Tensor
        self.rff_proj = nn.Linear(2 * n_freqs, emb_dim)

        ## prediction head
        self.token_head = nn.Linear(emb_dim, num_tokens)
        self.time_head = nn.Linear(emb_dim, vocab.time_bins)

    def embed(self, ids: Int[Tensor, "B N"]) -> Float[Tensor, "B N D"]:
        positions = th.clamp(ids - self.t0, min=0) # B N
        thetas = positions[:,:,None] * self.rff_freqs[None,None]  # [B, N, n_freqs]
        features = th.cat([th.sin(thetas), th.cos(thetas)], dim=-1)
        time_embs = self.rff_proj(features) # B N D

        return th.where(
            (ids >= self.t0)[:,:,None],
            time_embs,
            self.token_emb(th.clamp(ids, max=self.t0)),
        )
    
    def logits(self, pred_embs: Float[Tensor, "B N D"]) -> Float[Tensor, "B N V"]:
        return self._denormalized_logits(
            self.token_head(pred_embs),
            self.time_head(pred_embs),
        )
        
    def _denormalized_logits(
        self,
        token_logits: Float[Tensor, "B N D"],
        time_logits: Float[Tensor, "B N T"],
    ) -> Float[Tensor, "B N V"]:
        token_probs = token_logits.softmax(dim=-1)

        # P(time_i) = P(time_type) * P(time_i | time_type)
        marginal_time_probs = token_probs[:,:,-1:] * time_logits.softmax(dim=-1)

        combined_probs = th.cat([token_probs[:,:,:-1], marginal_time_probs], dim=-1)
        return th.log(combined_probs + 1e-8)

    def forward(
        self,
        pred_embs: Float[Tensor, "B N D"],
        true_tokens: Int[Tensor, "B N"],
        calc_accuracy: bool = False,
    ) -> tuple[
        Float[Tensor, ""],  # loss
        dict[str, float],   # log dict
    ]:
        is_time = true_tokens >= self.t0

        pred_token_logits = self.token_head(pred_embs)
        true_token_classes = th.clamp(true_tokens, max=self.t0)
        token_loss = F.cross_entropy(
            pred_token_logits.reshape(-1, pred_token_logits.size(-1)), 
            true_token_classes.reshape(-1),
        )

        # continuous ranked probability loss
        pred_time_logits = self.time_head(pred_embs)
        pred_timing_cdf = pred_time_logits.softmax(dim=-1).cumsum(dim=-1)
        true_time_classes = th.clamp(true_tokens - self.t0, min=0)
        domain = th.arange(pred_time_logits.size(-1)).to(true_tokens)
        true_timing_cdf = (domain >= true_time_classes[:,:,None]).float()
        time_loss = F.mse_loss(
            pred_timing_cdf[is_time],
            true_timing_cdf[is_time],
        )

        loss = token_loss + self.time_loss_factor * time_loss
        log_dict = {
            "token": token_loss.detach().item(),
            "time": time_loss.detach().item(),
            "loss": loss.detach().item(),
        }

        if calc_accuracy:
            with th.no_grad():
                pred_logits = self._denormalized_logits(pred_token_logits, pred_time_logits)
                pred_tokens = pred_logits.argmax(dim=-1)
                accuracy = (pred_tokens == true_tokens).float().mean()
                log_dict["accuracy"] = accuracy.item()

        return loss, log_dict