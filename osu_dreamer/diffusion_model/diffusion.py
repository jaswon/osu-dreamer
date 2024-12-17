
from collections.abc import Callable
from jaxtyping import Float

from dataclasses import dataclass

import torch as th
from torch import Tensor, nn

from einops import repeat

import osu_dreamer.modules.mp as MP

B = Float[Tensor, "B"]              # noise level, loss weight, uncertainty
F = Float[Tensor, "B F"]            # noise level features
X = Float[Tensor, "B D N"]          # sequence
Denoiser = Callable[[ X, B ], X]    # p(x0 | xt, t)

@dataclass
class DiffusionArgs:
    log_snr_scale: float
    log_snr_bound: float
    std_data: float
    noise_level_features: int


class Diffusion(nn.Module):
    """https://arxiv.org/pdf/2206.00364.pdf"""

    def __init__(
        self,
        args: DiffusionArgs,
    ):
        super().__init__()
        def std_noise(t):
            t = .5 + (t-.5) * (1-args.log_snr_bound*2) # [0,1] -> [b,1-b]
            log_snr = args.log_snr_scale * th.sign(0.5 - t) * th.log(1 - 2 * th.abs(0.5 - t)) # laplace 
            return args.std_data * th.exp(-.5 * log_snr)
        self.std_noise = std_noise
        self.std_data = args.std_data

        self.noise_level_features = MP.RandomFourierFeatures(1, args.noise_level_features)
        self.proj_u = MP.Linear(args.noise_level_features, 1)

    def pred_x0(self, model: Denoiser, x_t: X, std: B) -> tuple[X,B]:
        """https://arxiv.org/pdf/2206.00364.pdf#section.5"""

        sq_sum = std ** 2 + self.std_data ** 2
        hyp = sq_sum.sqrt()
        c_skip = self.std_data ** 2 / sq_sum
        c_out = std * self.std_data / hyp
        c_in = 1 / hyp
        c_noise = th.log(std) / 4

        nlf = self.noise_level_features(c_noise[:,None])
        u = self.proj_u(nlf)[:,0]
        model_out = model(c_in[:,None,None] * x_t, nlf)
        pred_x0 = c_skip[:,None,None] * x_t + c_out[:,None,None] * model_out
        return pred_x0, u

    def training_sample(self, model: Denoiser, x0: X) -> tuple[X,B,B]:
        """sample denoised predictions and per-batch loss weights"""
        b = x0.size(0)
        u = (th.arange(b) + th.rand(1)) / b # low discrepancy sampling
        t = self.std_noise(u).to(x0.device)
        loss_weight = (t ** 2 + self.std_data ** 2) / (t * self.std_data) ** 2
        x_t = x0 + th.randn_like(x0) * t[:,None,None]

        pred_x0, u = self.pred_x0(model, x_t, t)
        return pred_x0, u, loss_weight
    
    @th.no_grad()
    def sample(
        self, 
        denoiser: Denoiser,
        num_steps: int,
        z: X,

        show_progress: bool = False,
        S_churn: float = 40.,
        S_tmin: float = 1e-1,
        S_tmax: float = 1e1,
        S_noise: float = 1.003,
    ) -> X:
        """https://github.com/NVlabs/edm/blob/62072d2612c7da05165d6233d13d17d71f213fee/generate.py#L25"""
        
        sigmas = self.std_noise(th.linspace(0, 1, num_steps))
        num_churn_steps = ((S_tmin <= sigmas) & (sigmas <= S_tmax)).sum().item()
        sigmas = th.tensor([*sigmas.tolist(), 0], device=z.device)

        loop = zip(sigmas[:-1], sigmas[1:])
        if show_progress:
            from tqdm import tqdm
            loop = tqdm(loop, total=num_steps)

        def dx_dt(x: X, t: Float[Tensor, ""]):
            """
            https://arxiv.org/pdf/2011.13456.pdf#section.5
            
            score: ∇logp(xt|y) 
                = ∇log[p(xt) * p(y|xt)] 
                = ∇logp(xt) + ∇logp(y|xt)
                = (pred_x0 - xt) / t**2 + ∇logp(y|xt)
            """
            pred_x0, _ = self.pred_x0(denoiser, x, repeat(t, '-> b', b=x.size(0)))
            score = (pred_x0 - x) / t ** 2
            return -t * score

        x_t = z * (sigmas[0] ** 2 + self.std_data ** 2) ** .5
        for t_cur, t_nxt in loop:
            # increase noise temporarily
            gamma = min(S_churn / num_churn_steps, 2**.5 - 1) if S_tmin <= t_cur <= S_tmax else 0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_t + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * th.randn_like(x_t)

            # euler step
            d_cur = dx_dt(x_hat, t_hat)
            x_t = x_hat + (t_nxt - t_hat) * d_cur

            # 2nd order correction (Huen's method)
            if t_nxt > 0:
                d_prime = dx_dt(x_t, t_nxt)
                x_t = x_hat + 0.5 * (t_nxt - t_hat) * (d_cur + d_prime)

        return x_t
