
from jaxtyping import Float, Int, Bool

import torch as th
from torch import nn, Tensor

from osu_dreamer.lm.data.tokens.tokens import Vocab

from .token_head import TokenHead
from .time_head import TimeHead
from .pos_head import PositionHead

class DecoderHead(nn.Module):
    def __init__(
        self,
        vocab: Vocab,
        emb_dim: int,
        h_dim: int,
        time_loss_factor: float = 1.,
        pos_loss_factor: float = 1.,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.time_loss_factor = time_loss_factor
        self.pos_loss_factor = pos_loss_factor
        self.vocab = vocab

        self.token_head = TokenHead(vocab, emb_dim)
        self.pos_head = PositionHead(vocab, emb_dim, h_dim)
        self.time_head = TimeHead(vocab, emb_dim)

    def embed(self, seq: Int[Tensor, "*B 4"]) -> Float[Tensor, "*B D"]:
        embs = self.token_head.embed(seq[...,0]) # *B D

        is_pos = seq[...,0] == self.vocab.POS
        if is_pos.any():
            embs[is_pos] = self.pos_head.embed(seq[...,[1,2]][is_pos]).type_as(embs)

        is_time = seq[...,0] == self.vocab.TIME
        if is_time.any():
            embs[is_time] = self.time_head.embed(seq[...,3][is_time]).type_as(embs)
        
        return embs

    def forward(
        self,
        pred_embs: Float[Tensor, "*B D"],
        true_seq: Int[Tensor, "*B 4"],
        validation: bool = False,
    ) -> tuple[
        Float[Tensor, ""],  # loss
        dict[str, float],   # log dict
    ]:

        token_loss, token_loss_dict = self.token_head(
            pred_embs, 
            true_seq[...,0], 
            validation=validation,
        )

        is_pos = true_seq[...,0] == self.vocab.POS
        if is_pos.any():
            pos_loss, pos_loss_dict = self.pos_head(
                pred_embs[is_pos], 
                true_seq[...,[1,2]][is_pos], 
                validation=validation,
            )
        else:
            pos_loss, pos_loss_dict = th.tensor(0, device=pred_embs.device), {}

        is_time = true_seq[...,0] == self.vocab.TIME
        if is_time.any():
            time_loss, time_loss_dict = self.time_head(
                pred_embs[is_time], 
                true_seq[...,3][is_time], 
                validation=validation,
            )
        else:
            time_loss, time_loss_dict = th.tensor(0, device=pred_embs.device), {}

        loss = token_loss + self.pos_loss_factor * pos_loss + self.time_loss_factor * time_loss
        return loss, {
            **{ f"token/{k}":v for k,v in token_loss_dict.items() },
            **{ f"pos/{k}":v for k,v in pos_loss_dict.items() },
            **{ f"time/{k}":v for k,v in time_loss_dict.items() },
            "loss/total": loss.detach().item(),
        }
    
    @th.no_grad()
    def sample(
        self,
        embs: Float[Tensor, "D"],
        mask_ids: Bool[Tensor, "V"],
        mask_before_time: int,
        top_p: float,
    ) -> Int[Tensor, "4"]:
        return th.cat([
            self.token_head.sample(embs, top_p, mask_ids),
            self.pos_head.sample(embs, 4,4, top_p),
            self.time_head.sample(embs, top_p, th.tensor(mask_before_time, device=embs.device)),
        ], dim=-1)