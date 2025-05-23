
from jaxtyping import Float, Int

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from .data.tokens import VOCAB_SIZE, PAD

def focal_loss(
    inputs: Float[Tensor, "B D ..."],
    target: Int[Tensor, "B ..."],
    gamma: float,
    weight: None | Float[Tensor, "D"] = None,
    *args, **kwargs,
) -> Float[Tensor, "B ..."]:
    logpt = F.log_softmax(inputs, dim=1)
    inputs = (1 - logpt.exp()).pow(gamma) * logpt
    return F.nll_loss(inputs, target, weight, reduction='none', *args, **kwargs)


def kumaraswamy_cdf(params: Float[Tensor, "... 2"], n_points: int) -> Float[Tensor, "... N"]:
    from torch.distributions import Kumaraswamy
    dist = Kumaraswamy(1+params[...,:1], 1+params[...,1:])
    points = th.linspace(0, 1, n_points, device=params.device).view(*([1] * (params.ndim-1)), -1)
    return dist.cdf(points)

def kumaraswamy_logits(params: Float[Tensor, "... 2"], n_points: int) -> Float[Tensor, "... N"]:
    from torch.distributions import Kumaraswamy
    dist = Kumaraswamy(1+params[...,:1], 1+params[...,1:])
    points = th.linspace(0, 1, n_points, device=params.device).view(*([1] * (params.ndim-1)), -1)
    return th.nan_to_num(dist.log_prob(points), nan=-th.inf)

class ModalHead(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        timing_domain: int,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.timing_domain = timing_domain

        # 3 modes: token, timing, position
        self.mode_head = nn.Linear(emb_dim, 3)

        # token head
        self.token_emb = nn.Embedding(VOCAB_SIZE, emb_dim)
        th.nn.init.normal_(self.token_emb.weight)
        self.token_head = nn.Linear(emb_dim, VOCAB_SIZE)
        self.token_head.weight = self.token_emb.weight

        # timing head
        self.timing_emb = nn.Embedding(timing_domain, emb_dim) # ... -> ... E
        th.nn.init.normal_(self.timing_emb.weight)
        self.timing_head = nn.Sequential(
            nn.Linear(emb_dim, 2),
            nn.Softplus(),
        ) # parametrize timing distribution via kumaraswamy

        # position head
        self.pos_emb = nn.Linear(2, emb_dim) # ... 2 -> ... E
        self.pos_head = nn.Linear(emb_dim, 2) # ... E -> ... 2
        
    def forward(
        self,
        pred_embs: Float[Tensor, "... E"],
        true_modes: Int[Tensor, "..."],
        true_tokens: Int[Tensor, "..."],
        true_timings: Int[Tensor, "..."],
        true_positions: Float[Tensor, "... 2"],
        focal_gamma: float = 1.,
    ) -> tuple[
        Float[Tensor, ""],              # loss
        dict[str, Float[Tensor, ""]],   # log dict
    ]:
        D = pred_embs.device
        
        # mode loss - focal
        pred_modes = self.mode_head(pred_embs) # ... 3
        mode_loss = focal_loss(
            pred_modes.flatten(0,-2),
            true_modes.flatten(),
            gamma = focal_gamma,
        ).mean()

        # token loss - focal
        pred_token_logits = self.token_head(pred_embs) # ... V
        token_loss = focal_loss(
            pred_token_logits[true_modes == 0],
            true_tokens[true_modes == 0],
            gamma = focal_gamma,
            ignore_index = PAD,
        ).mean()

        # timing loss - crps
        pred_timing_dist_params = self.timing_head(pred_embs) # ... 2
        pred_timing_cdf = kumaraswamy_cdf(pred_timing_dist_params, self.timing_domain) # ... S
        true_timing_cdf = (th.arange(self.timing_domain).to(D) >= true_timings[...,None]).float() # ... S
        timing_loss = F.mse_loss(
            pred_timing_cdf[true_modes == 1],
            true_timing_cdf[true_modes == 1],
        )

        # position loss - mse
        pred_positions = self.pos_head(pred_embs) # ... 2
        position_loss = F.mse_loss( 
            pred_positions[true_modes == 2],
            true_positions[true_modes == 2],
        )

        loss = mode_loss + token_loss + timing_loss + position_loss
        return loss, {
            "loss": loss.detach(),
            "mode": mode_loss.detach(),
            "token": token_loss.detach(),
            "timing": timing_loss.detach(),
            "position": position_loss.detach(),
        }
    
    def embed(
        self,
        modes: Int[Tensor, "..."],
        tokens: Int[Tensor, "..."],
        timings: Int[Tensor, "..."],
        positions: Float[Tensor, "... 2"],
    ) -> Float[Tensor, "... E"]:
        return th.gather(
            th.stack([
                self.token_emb(tokens),
                self.timing_emb(timings),
                self.pos_emb(positions),
            ], dim=0), # K ... E
            dim = 0,
            index = modes[None,...,None].expand(-1,*((-1,) * modes.ndim),self.emb_dim),
        )[0]

    def sample(self, embs: Float[Tensor, "B E"], timings_to_mask: Int[Tensor, "B"]) -> tuple[
        Int[Tensor, "B"],       # modes
        Int[Tensor, "B"],       # tokens
        Int[Tensor, "B"],       # timings
        Float[Tensor, "B 2"]    # positions
    ]:
        # sample modes
        mode_logits = self.mode_head(embs) # B 3
        modes = th.multinomial(mode_logits.softmax(dim=-1), num_samples=1)[:,0] # B

        # sample tokens
        token_logits = self.token_head(embs) # B V
        tokens = th.multinomial(token_logits.softmax(dim=-1), num_samples=1)[:,0] # B
        tokens[modes != 0] = PAD

        # sample timings
        timing_dist_params = self.timing_head(embs) # ... 2
        timing_logits = kumaraswamy_logits(timing_dist_params, self.timing_domain) # ... S
        for b_idx, num_mask in enumerate(timings_to_mask):
            timing_logits[b_idx, :num_mask] = -th.inf
        timings = th.multinomial(timing_logits.softmax(dim=-1), num_samples=1)[:,0] # B
        timings[modes != 1] = 0

        # sample positions
        positions = self.pos_head(embs).float() # B 2
        positions[modes != 2] = 0

        return modes, tokens, timings, positions