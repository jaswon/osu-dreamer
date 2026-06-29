
from jaxtyping import Float

import torch as th
from torch import Tensor

def sigreg_weak_loss(x: Float[Tensor, "N D"], sketch_dim=64) -> Float[Tensor, ""]:
    """
    Forces Covariance(x) ~ Identity via Sketching.
    Approximates the 2nd Moment constraint efficiently.
    """
    N,D = x.size()

    # 1. Sketching (Dimensionality Reduction)
    if D > sketch_dim:
        # Random projection preserves geometric structure (Johnson-Lindenstrauss)
        x = x @ th.randn(D, sketch_dim, device=x.device) / (D ** 0.5)
    else:
        sketch_dim = D

    # 2. Centering & Covariance
    x = x - x.mean(dim=0, keepdim=True)
    cov = (x.T @ x) / (N - 1 + 1e-6)

    # 3. Target Identity & Loss
    target = th.eye(sketch_dim, device=x.device)

    # Minimize Frobenius norm distance to Identity
    return th.linalg.matrix_norm(cov - target, ord='fro')