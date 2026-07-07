
import torch as th

def mmd_imq(z: th.Tensor, z_prior: th.Tensor) -> th.Tensor:
    """
    unbiased MMD^2 between encoded samples `z` and prior samples `z_prior`,
    both shaped (N, E), using a sum of inverse-multiquadratic kernels.
    this is the WAE-MMD regularizer (Tolstikhin et al. 2018): it pushes the
    aggregated posterior towards the N(0, I) prior without the per-sample KL
    (and reparameterization) that a VAE requires.
    """
    n, d = z.shape
    C_base = 2. * d
    scales = (.1, .2, .5, 1., 2., 5., 10.)

    def kernel(a: th.Tensor, b: th.Tensor) -> th.Tensor:
        d2 = th.cdist(a, b).pow(2)
        out = th.zeros_like(d2)
        for s in scales:
            C = C_base * s
            out = out + C / (C + d2)
        return out

    off_diag = 1. - th.eye(n, device=z.device, dtype=z.dtype)
    zz = (kernel(z, z) * off_diag).sum() / (n * (n - 1))
    pp = (kernel(z_prior, z_prior) * off_diag).sum() / (n * (n - 1))
    zp = kernel(z, z_prior).mean()
    return zz + pp - 2. * zp