
from typing import Optional
from collections.abc import Callable
from jaxtyping import Float

import numpy as np

import torch as th
from torch import Tensor
import torch.nn.functional as F

from einops import repeat


T = Float[Tensor, "B 1 1"]          # diffusion step
X = Float[Tensor, "B C L"]          # sequence
Model = Callable[[ X, X, T ], X]    # p(x0 | p(x0 | xu, u), xt, t)

class Diffusion:
    """https://arxiv.org/pdf/2206.00364.pdf"""

    def __init__(
        self,
        P_mean: float,
        P_std: float,
        std_data: float = .5,
    ):
        super().__init__()

        self.std_data = std_data
        self.sample_t = lambda *shape: th.exp(P_mean + th.randn(shape) * P_std)

    def pred_x0(self, model: Model, y: X, x_t: X, std: T) -> X:
        """https://arxiv.org/pdf/2206.00364.pdf#section.5"""

        sq_sum = std ** 2 + self.std_data ** 2
        hyp = sq_sum.sqrt()
        c_skip = self.std_data ** 2 / sq_sum
        c_out = std * self.std_data / hyp
        c_in = 1 / hyp
        c_noise = th.log(std)[:,0,0]

        pred_x0 = c_skip * x_t + c_out * model(y, c_in * x_t, c_noise)
        return pred_x0.clamp(min=-1, max=1)

    def loss(self, model: Model, x0: X) -> Float[Tensor, ""]:
        """sample denoised predictions of training data for denoising score matching objective"""

        t = self.sample_t(x0.size(0),1,1).to(x0.device)
        loss_weight = (t ** 2 + self.std_data ** 2) / (t * self.std_data) ** 2
        x_t = x0 + th.randn_like(x0) * t

        x0_hat_uncond = self.pred_x0(model, th.zeros_like(x0), x_t, t)
        x0_hat_cond = self.pred_x0(model, x0_hat_uncond.detach(), x_t, t)
        x0_hat = th.stack([ x0_hat_uncond, x0_hat_cond ], dim=0)

        return (loss_weight * ( x0_hat - x0 ) ** 2).mean()
    
    @th.no_grad()
    def sample(
        self, 
        denoiser: Model, 
        guide: Optional[Callable[[X], X]],
        num_steps: int,
        z: X,
        show_progress: bool = False,

        t_min: float = .002,
        t_max: float = 80.,
        rho: float = 7.,
        S_churn: float = 10.,
        S_min: float = 0,
        S_max: float = float('inf'),
        S_noise: float = 1.003,
    ) -> X:
        """https://github.com/NVlabs/edm/blob/62072d2612c7da05165d6233d13d17d71f213fee/generate.py#L25"""

        t = th.linspace(1, 0, num_steps)
        sigmas = ( t_min ** (1/rho) + t * (t_max ** (1/rho) - t_min ** (1/rho)) ) ** rho
        sigmas = th.tensor([*sigmas.tolist(), 0], device=z.device)
        sigmas = repeat(sigmas, 's -> s b 1 1', b = z.size(0))

        x_t = z * t_max

        loop = zip(sigmas[:-1], sigmas[1:])
        if show_progress:
            from tqdm import tqdm
            loop = tqdm(loop, total=num_steps)

        def compute_score(pred_x0, x, t):
            """
            https://arxiv.org/pdf/2011.13456.pdf#section.5
            
            score: ∇logp(xt|y,t) = ∇logp(xt|t) + ∇logp(y|xt)
            """
            pred_x0 = self.pred_x0(denoiser, pred_x0, x, t)
            if guide is not None:
                with th.enable_grad():
                    guide_score = F.logsigmoid(guide(pred_x0.requires_grad_())).mean()
                guidance = th.autograd.grad(guide_score, pred_x0)[0]
            else:
                guidance = 0
            return pred_x0, (pred_x0 - x) / t ** 2 - guidance

        pred_x0 = th.zeros_like(x_t)
        for t_cur, t_nxt in loop:

            # increase noise temporarily
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur[0,0,0] <= S_max else 0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_t + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * th.randn_like(x_t)

            # euler step
            pred_x0, score = compute_score(pred_x0, x_hat, t_hat)
            d_cur = -t_hat * score
            x_t = x_hat + (t_nxt - t_hat) * d_cur

            # 2nd order correction (Huen's method)
            if t_nxt[0,0,0] > 0:
                pred_x0, score = compute_score(pred_x0, x_t, t_nxt)
                d_prime = -t_nxt * score
                x_t = x_hat + (t_nxt - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_t
