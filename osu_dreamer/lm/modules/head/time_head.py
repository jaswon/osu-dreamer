
from jaxtyping import Float, Int

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from osu_dreamer.lm.data.tokens.tokens import Vocab

from .sample import sample

class TimeHead(nn.Module):
    def __init__(
        self, 
        vocab: Vocab,
        emb_dim: int, 
    ):
        super().__init__()
        self.n_classes = 1+vocab.time_bins # last class is EOS
        self.emb = nn.Embedding(self.n_classes, emb_dim)
        self.head = nn.Linear(emb_dim, self.n_classes)

    def embed(self, i: Int[Tensor, "*B"]) -> Float[Tensor, "*B D"]:
        assert (i>=0).all()
        assert (i<self.n_classes).all()
        return self.emb(i)

    def forward(
        self,
        pred_embs: Float[Tensor, "B D"],
        true_classes: Int[Tensor, "B"],
        validation: bool = False,
    ) -> tuple[
        Float[Tensor, ""],  # loss
        dict[str, float],   # log dict
    ]:
        logits = self.head(pred_embs)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), 
            true_classes.reshape(-1),
        )
        log_dict = { "loss": loss.detach().item() }

        if validation:
            with th.no_grad():
                log_dict["val/mae"] = (logits.argmax(dim=-1) - true_classes).abs().float().mean().item()

                log_probs = logits.log_softmax(dim=-1)
                log_dict["val/entropy"] = (log_probs.exp() * -log_probs).sum(dim=-1).mean().item()
                log_dict["val/nll"] = -log_probs[th.arange(true_classes.size(0)), true_classes].mean().item()

        return loss, log_dict
    
    @th.no_grad()
    def sample(
        self,
        embs: Float[Tensor, "*B D"],
        top_p: float,
        mask_up_to: None | Int[Tensor, "*B"] = None, 
    ) -> Int[Tensor, "*B 1"]:
        
        logits = self.head(embs)

        if mask_up_to is not None:
            assert (mask_up_to >= 0).all()
            logits[th.arange(logits.size(-1), device=mask_up_to.device) <= mask_up_to[...,None]] = -th.inf

        return sample(logits.softmax(dim=-1), top_p)