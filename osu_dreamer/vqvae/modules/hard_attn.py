
from jaxtyping import Float, Int

from dataclasses import dataclass

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from einops import rearrange

@dataclass
class HardAttnArgs:
    num_codes: int
    num_heads: int
    temperature: float = 1.

class HardAttn(nn.Module):
    """https://arxiv.org/abs/2106.04283"""
    
    def __init__(
        self,
        dim: int,
        args: HardAttnArgs,
    ):
        super().__init__()
        head_dim = dim // args.num_heads
        assert head_dim * args.num_heads == dim
        self.head_dim = head_dim
        self.num_heads = args.num_heads
        self.num_codes = args.num_codes
        self.register_buffer('temperature', th.tensor(args.temperature)); self.temperature: Tensor

        self.kv = nn.Parameter(th.randn(2, args.num_heads, args.num_codes, head_dim) * head_dim**.5)
    
    def compute_perplexity(
        self,
        indices: Int[Tensor, "B H L"],
    ) -> float:
        indices = rearrange(indices, 'b h l -> (b l) h')
        powers = th.arange(indices.size(1), device=indices.device)
        ids = (indices * self.num_codes**powers).sum(dim=1)
        _, counts = th.unique(ids, return_counts=True)
        probs = counts / ids.numel()
        return th.exp(-th.sum(probs * th.log(probs.clamp(min=1e-6)))).item()

    def forward(
        self, 
        x: Float[Tensor, "B X L"],
    ) -> tuple[
        Float[Tensor, "B X L"], # hard attention
        Float[Tensor, ""],      # entropy
        Int[Tensor, "B H L"],   # codebook indices
    ]:
        q = F.rms_norm(rearrange(x, 'b (h d) l -> b h l d', h=self.num_heads), (self.head_dim,))
        k,v = F.rms_norm(self.kv, (self.head_dim,))
        logits = th.einsum('bhnd,hmd->bhnm', q, k)

        posterior_dist = th.distributions.Categorical(logits=logits)
        entropy = posterior_dist.entropy().mean()

        if self.training:
            # softmax for gradient stability (Appendix B.1.5)
            attn = F.gumbel_softmax(logits, tau=self.temperature.item(), dim=-1, hard=True)
            codebook_indices = attn.argmax(dim=-1)
            self.temperature = (self.temperature * 0.9999).clamp(min=.1)
        else:
            codebook_indices = logits.argmax(dim=-1)
            attn = F.one_hot(codebook_indices, num_classes=self.num_codes).float()

        out = th.einsum('bhnm,hmd->bhnd', attn, v)
        
        return rearrange(out, 'b h l d -> b (h d) l'), entropy, codebook_indices