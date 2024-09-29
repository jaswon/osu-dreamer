
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
        P_std: float,
        std_data: float = 1.,
    ):
        super().__init__()

        self.std_data = std_data

        def sample_t(x: X) -> T:
            return std_data * th.exp(P_std * th.randn(x.size(0),1,1).to(x.device))
        self.sample_t = sample_t

    def pred_x0(self, model: Denoiser, x_t: X, std: T) -> X:
        """https://arxiv.org/pdf/2206.00364.pdf#section.5"""

        sq_sum = std ** 2 + self.std_data ** 2
        hyp = sq_sum.sqrt()
        c_skip = self.std_data ** 2 / sq_sum
        c_out = std * self.std_data / hyp
        c_in = 1 / hyp
        c_noise = th.log(std)[:,0,0]

        return c_skip * x_t + c_out * model(c_in * x_t, c_noise)

    def training_sample(self, model: Denoiser, x0: X) -> tuple[X,T]:
        """sample denoised predictions and per-batch loss weights"""
        t = self.sample_t(x0)
        loss_weight = (t ** 2 + self.std_data ** 2) / (t * self.std_data) ** 2
        x_t = x0 + th.randn_like(x0) * t

        return self.pred_x0(model, x_t, t), loss_weight
    
    @th.no_grad()
    def sample(
        self, 
        denoiser: Denoiser,
        num_steps: int,
        z: X,

        show_progress: bool = False,
        snr_scale: float = 2.5,
        snr_offset: float = 1e-1,
        S_churn: float = 40.,
        S_tmin: float = 1e-1,
        S_tmax: float = 1e1,
        S_noise: float = 1.003,
    ) -> X:
        """https://github.com/NVlabs/edm/blob/62072d2612c7da05165d6233d13d17d71f213fee/generate.py#L25"""
        
        t = th.linspace(snr_offset, 1, num_steps+1, device=z.device)
        log_snr = snr_scale * th.sign(0.5 - t) * th.log(1 - 2 * th.abs(0.5 - t)) # laplace 
        sigmas = th.exp(-.5 * log_snr)
        sigmas = repeat(sigmas, 's -> s b 1 1', b = z.size(0))

        loop = zip(sigmas[:-1], sigmas[1:])
        if show_progress:
            from tqdm import tqdm
            loop = tqdm(loop, total=num_steps)

        def dx_dt(x: X, t: T):
            """
            https://arxiv.org/pdf/2011.13456.pdf#section.5
            
            score: ∇logp(xt|y) 
                = ∇log[p(xt) * p(y|xt)] 
                = ∇logp(xt) + ∇logp(y|xt)
                = (pred_x0 - xt) / t**2 + ∇logp(y|xt)
            """
            x0_hat = self.pred_x0(denoiser, x, t)
            score = (x0_hat - x) / t ** 2
            return -t * score

        x_t = z * sigmas[0,0,0,0]
        for t_cur, t_nxt in loop:
            # increase noise temporarily
            gamma = min(S_churn / num_steps, 2**.5 - 1) if S_tmin <= t_cur[0,0,0] <= S_tmax else 0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_t + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * th.randn_like(x_t)

            # euler step
            d_cur = dx_dt(x_hat, t_hat)
            x_t = x_hat + (t_nxt - t_hat) * d_cur

            # 2nd order correction (Huen's method)
            if t_nxt[0,0,0] > 0:
                d_prime = dx_dt(x_t, t_nxt)
                x_t = x_hat + 0.5 * (t_nxt - t_hat) * (d_cur + d_prime)

        return x_t
