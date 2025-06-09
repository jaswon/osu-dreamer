
from jaxtyping import Float, Int

from dataclasses import dataclass

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import EinMix as Mix

@dataclass
class HardAttnArgs:
    num_codes: int
    code_dim: int
    num_heads: int = 8
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
        self.temperature = args.temperature

        self.codes = nn.Parameter(th.randn(args.num_codes, args.code_dim))
        self.to_k = Mix('n c -> h n d', 'c h d', c=args.code_dim, h=args.num_heads, d=head_dim)
        self.to_v = Mix('n c -> h n d', 'c h d', c=args.code_dim, h=args.num_heads, d=head_dim)
    
    def lookup(
        self,
        i: Int[Tensor, "B H L"],
    ) -> Float[Tensor, "B X L"]:
        v = self.to_v(self.codes) # H N D
        out = th.gather(
            repeat(v, 'h n d -> b h n d', b=i.size(0)),
            dim = 2,
            index = repeat(i, 'b h l -> b h l d', d=v.size(-1)),
        )
        return rearrange(out, 'b h l d -> b (h d) l')
    
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
        k = F.rms_norm(self.to_k(self.codes), (self.head_dim,))
        v = self.to_v(self.codes)
        logits = th.einsum('bhnd,hmd->bhnm', q, k)

        posterior_dist = th.distributions.Categorical(logits=logits)
        entropy = posterior_dist.entropy().mean()

        if self.training:
            # softmax for gradient stability (Appendix B.1.5)
            attn = F.gumbel_softmax(logits, tau=self.temperature, dim=-1, hard=True)
            codebook_indices = attn.argmax(dim=-1)
        else:
            codebook_indices = logits.argmax(dim=-1)
            attn = F.one_hot(codebook_indices, num_classes=self.codes.size(0)).float()

        out = th.einsum('bhnm,hmd->bhnd', attn, v)
        
        return rearrange(out, 'b h l d -> b (h d) l'), entropy, codebook_indices