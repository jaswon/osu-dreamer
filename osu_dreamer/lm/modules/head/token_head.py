
from jaxtyping import Float, Int, Bool

import torch as th
import torch.nn.functional as F
from torch import nn, Tensor

from osu_dreamer.lm.data.tokens.tokens import Vocab

from .sample import sample
    
class TokenHead(nn.Module):
    def __init__(self, vocab: Vocab, emb_dim: int):
        super().__init__()
        self.vocab = vocab
        self.n_classes = len(vocab.tokens)
        self.emb = nn.Embedding(self.n_classes, emb_dim)
        self.head = nn.Linear(emb_dim, self.n_classes)

    def embed(self, i: Int[Tensor, "*B"]) -> Float[Tensor, "*B D"]:
        assert (i>=0).all()
        assert (i<self.n_classes).all()
        return self.emb(i)

    def forward(
        self,
        pred_embs: Float[Tensor, "*B D"],
        true_classes: Int[Tensor, "*B"],
        validation: bool = False,
    ) -> tuple[
        Float[Tensor, ""],  # loss
        dict[str, float],   # log dict
    ]:
        pred_logits = self.head(pred_embs)
        loss = F.cross_entropy(
            pred_logits.reshape(-1, pred_logits.size(-1)), 
            true_classes.reshape(-1),
        )
        log_dict = { "loss": loss.detach().item() }

        if validation:
            with th.no_grad():
                top_preds = self.head(pred_embs).argsort(dim=-1, descending=True)
                for k in [1, 5, 10]:
                    top_k_matches = (top_preds[...,:k] == true_classes[...,None]).any(dim=-1)
                    log_dict[f"val/top_{k}"] = top_k_matches.float().mean().item()
    
        return loss, log_dict
    
    @th.no_grad()
    def sample(
        self,
        embs: Float[Tensor, "*B D"],
        top_p: float,
        allow_mask: None | Bool[Tensor, "*B V"] = None, 
    ) -> Int[Tensor, "*B 1"]:
        
        logits = self.head(embs)

        if allow_mask is not None:
            logits[~allow_mask] = -th.inf

        return sample(logits.softmax(dim=-1), top_p)