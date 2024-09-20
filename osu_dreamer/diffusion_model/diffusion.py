
from collections.abc import Callable
from jaxtyping import Float

import torch as th
from torch import Tensor

from einops import repeat


T = Float[Tensor, "B 1 1"]              # diffusion step
X = Float[Tensor, "B D N"]              # sequence
Denoiser = Callable[[ X, T ], X]        # p(x0 | xt, t)

class Diffusion:
    """https://arxiv.org/pdf/2206.00364.pdf"""

    def __init__(
        self,
        P_mean: float,
        P_std: float,
        std_data: float = .8,
    ):
        super().__init__()

        self.std_data = std_data
        self.tZ = lambda Z: th.exp(P_mean + Z * P_std)

    def pred_x0(self, model: Denoiser, x_t: X, std: T) -> X:
        """https://arxiv.org/pdf/2206.00364.pdf#section.5"""

        sq_sum = std ** 2 + self.std_data ** 2
        hyp = sq_sum.sqrt()
        c_skip = self.std_data ** 2 / sq_sum
        c_out = std * self.std_data / hyp
        c_in = 1 / hyp
        c_noise = th.log(std)[:,0,0]

        pred_x0 = c_skip * x_t + c_out * model(c_in * x_t, c_noise)
        return pred_x0.clamp(min=-1, max=1)

    def loss(self, model: Denoiser, x0: X) -> Float[Tensor, ""]:
        """sample denoised predictions of training data for denoising score matching objective"""
        t = self.tZ(th.randn(x0.size(0),1,1)).to(x0.device)
        loss_weight = (t ** 2 + self.std_data ** 2) / (t * self.std_data) ** 2
        x_t = x0 + th.randn_like(x0) * t

        return th.mean( loss_weight * ( self.pred_x0(model, x_t, t) - x0 ) ** 2 )
    
    @th.no_grad()
    def sample(
        self, 
        denoiser: Denoiser,
        num_steps: int,
        z: X,

        show_progress: bool = False,
        s_min: float = .001,
        s_max: float = 50.,
        rho: float = 7.,
        S_churn: float = 0.,
        S_tmin: float = 0.05,
        S_tmax: float = 50.,
        S_noise: float = 1.003,
    ) -> X:
        """https://github.com/NVlabs/edm/blob/62072d2612c7da05165d6233d13d17d71f213fee/generate.py#L25"""

        sigmas = th.linspace(s_max ** (1/rho), s_min ** (1/rho), num_steps) ** rho
        sigmas = th.tensor([*sigmas.tolist(), 0], device=z.device)
        sigmas = repeat(sigmas, 's -> s b 1 1', b = z.size(0))

        loop = zip(sigmas[:-1], sigmas[1:])
        if show_progress:
            from tqdm import tqdm
            loop = tqdm(loop, total=num_steps)

        def compute_score(x, t):
            """
            https://arxiv.org/pdf/2011.13456.pdf#section.5
            
            score: ∇logp(xt|y) 
                = ∇log[p(xt) * p(y|xt)] 
                = ∇logp(xt) + ∇logp(y|xt)
                = (pred_x0 - xt) / t**2 + ∇logp(y|xt)
            """
            x0_hat = self.pred_x0(denoiser, x, t)
            return (x0_hat - x) / t ** 2

        x_t = z * s_max
        for t_cur, t_nxt in loop:
            # increase noise temporarily
            gamma = min(S_churn / num_steps, 2**.5 - 1) if S_tmin <= t_cur[0,0,0] <= S_tmax else 0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_t + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * th.randn_like(x_t)

            # euler step
            d_cur = -t_hat * compute_score(x_hat, t_hat)
            x_t = x_hat + (t_nxt - t_hat) * d_cur

            # 2nd order correction (Huen's method)
            if t_nxt[0,0,0] > 0:
                d_prime = -t_nxt * compute_score(x_t, t_nxt)
                x_t = x_hat + 0.5 * (t_nxt - t_hat) * (d_cur + d_prime)

        return x_t
