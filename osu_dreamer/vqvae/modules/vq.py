
from jaxtyping import Float, Int

from dataclasses import dataclass

import torch as th
from torch import nn, Tensor
import torch.nn.functional as F

from einops import rearrange

@dataclass
class VQArgs:
    num_codes: int
    num_heads: int
    commitment_cost: float = 0.25
    decay: float = 0.99

class ProductQuantizer(nn.Module):
    def __init__(
        self,
        dim: int,
        args: VQArgs,
    ):
        super().__init__()
        self.num_codes = args.num_codes
        self.num_heads = args.num_heads
        self.head_dim = dim // args.num_heads
        assert self.head_dim * args.num_heads == dim, "dim must be divisible by num_heads"
        
        self.commitment_cost = args.commitment_cost
        self.decay = args.decay

        self.rms_norm = nn.RMSNorm([self.head_dim], elementwise_affine=False)
        self.register_buffer('codebooks', self.rms_norm(th.randn(args.num_heads, args.num_codes, self.head_dim))); self.codebooks: Tensor

        # TODO: remove
        self.register_buffer('codebook_usage', th.zeros(args.num_heads, args.num_codes)); self.codebook_usage: Tensor
        
    def forward(
        self, 
        z: Float[Tensor, "B D L"],
    ) -> tuple[
        Float[Tensor, "B D L"], # quantized
        Float[Tensor, ""],      # loss
        Int[Tensor, "B H L"],   # indices for each head
    ]:
        z_split = rearrange(z, 'b (h d) l -> h (b l) d', h=self.num_heads) # H N D
        indices = th.einsum('hnd,hmd->hnm', z_split, self.codebooks).argmax(dim=-1) # H N
        z_split_q = self.codebooks[th.arange(self.num_heads)[:,None], indices] # H N D

        commitment_loss = F.mse_loss(z_split.detach(), z_split_q)
        codebook_loss = F.mse_loss(z_split, z_split_q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Update codebook
        if self.training and self.decay != 0:
            with th.no_grad():
                for head, head_indices in enumerate(indices):
                    uniq_codes, uniq_code_locs = th.unique(head_indices, return_inverse=True)
                    for code_idx, code in enumerate(uniq_codes):
                        z_sel = z_split[head,uniq_code_locs==code_idx] # U D
                        if len(z_sel) > 0:
                            self.codebooks[head, code] = self.rms_norm(slerp(
                                z_sel.mean(dim=0),
                                self.codebooks[head, code],
                                self.decay,
                            ))
        
        z_q = rearrange(z_split_q, 'h (b l) d -> b (h d) l', b=z.size(0))
        indices = rearrange(indices, 'h (b l) -> b h l', b=z.size(0))
        
        z_q = z + (z_q - z).detach() # straight-through gradient estimation
        return z_q, vq_loss, indices
    

def slerp(
    low: Float[Tensor, "*B D"], 
    high: Float[Tensor, "*B D"],
    val: float,
) -> Float[Tensor, "*B D"]:
    omega = th.acos(F.cosine_similarity(low, high, dim=-1)).unsqueeze(-1)
    return (
        + low * th.sin((1.0-val)*omega)
        + high * th.sin(val*omega)
    ) / th.sin(omega)