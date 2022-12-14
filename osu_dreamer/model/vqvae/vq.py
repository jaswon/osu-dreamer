import torch
from torch import nn

import warnings
from scipy.cluster.vq import kmeans2

# taken from
# https://github.com/CompVis/taming-transformers/blob/24268930bf1dce879235a7fddd0b2355b84d7ea6/taming/modules/vqvae/quantize.py
class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta=0.1):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z: "B,C,L") -> "B,C,L":
        z: "B,L,C" = z.permute(0,2,1).contiguous()
        z_flattened: "BL,C" = z.reshape(-1, self.e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d: "BL,N" = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, self.embedding.weight.permute(1,0))

        inds: "BL," = torch.argmin(d, dim=1)
        z_q: "B,L,C" = self.embedding(inds).view(z.shape)

        # compute loss for embedding
        loss = self.beta * torch.mean((z_q.detach()-z)**2) + torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()
        
        z_q: "B,C,L" = z_q.permute(0,2,1).contiguous()

        return z_q, loss, inds