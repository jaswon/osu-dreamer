
from jaxtyping import Float, Int

from dataclasses import dataclass

import torch as th
import torch.nn.functional as F
from torch import nn, Tensor

import osu_dreamer.modules.mp as MP


@dataclass
class VQArgs:
    h_dim: int
    beta: float = .25

class VectorQuantizer(nn.Module):
    def __init__(
        self,
        n_e: int,
        e_dim: int,
        args: VQArgs,
    ):
        super().__init__()
        self.beta = args.beta
        self.n_e = n_e
        self.h_dim = args.h_dim
        self.norm = nn.RMSNorm(args.h_dim, elementwise_affine=False)
        self.proj_in = MP.Linear(e_dim, args.h_dim)
        self.codebook = nn.Parameter(th.randn(n_e, args.h_dim))
        self.proj_out = MP.Linear(args.h_dim, e_dim)

    def lookup(
        self,
        inds: Int[Tensor, "B L"],
    ) -> Float[Tensor, "B D L"]:
        codebook = self.norm(self.codebook)
        z_q = F.embedding(inds, codebook) # ... E
        return self.proj_out(z_q).transpose(1,2)

    def forward(
        self,
        z: Float[Tensor, "B D L"],
    ) -> tuple[
        Float[Tensor, "B D L"], # quantized embeddings
        Int[Tensor, "B L"],     # embedding indices
        Float[Tensor, ""],      # loss
    ]:
        b,_,l = z.size()
        z = self.proj_in(z.transpose(1,2).contiguous()) # B L E
        z = self.norm(z)
        z_flat = z.view(-1, self.h_dim) # BL E

        codebook = self.norm(self.codebook) # V E

        sim = ( z_flat[:,None] * codebook[None,:] ).sum(dim=-1) # BL N
        inds = th.argmax(sim, dim=1) # BL
        z_q = F.embedding(inds, codebook).view(z.shape) # B L E

        embedding_loss = (z_q.detach()-z).pow(2).mean()
        commit_loss = (z_q-z.detach()).pow(2).mean()
        loss = embedding_loss + self.beta * commit_loss
 
        z_q = z + (z_q - z).detach() # straight through gradient
        z_q = self.proj_out(z_q).transpose(1,2).contiguous()

        return z_q, inds.view(b,l), loss