
from collections.abc import Callable
from jaxtyping import Float

from dataclasses import dataclass

import torch as th
from torch import Tensor, nn

from einops import repeat

import osu_dreamer.modules.mp as MP

U = Float[Tensor, "B"]              # loss weight, uncertainty
F = Float[Tensor, "B F"]            # noise level features
T = Float[Tensor, "B 1 1"]          # diffusion step
X = Float[Tensor, "B D N"]          # sequence
Denoiser = Callable[[ X, T ], X]    # p(x0 | xt, t)

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

    def pred_x0(self, model: Denoiser, x_t: X, std: T) -> tuple[X,U]:
        """https://arxiv.org/pdf/2206.00364.pdf#section.5"""

        sq_sum = std ** 2 + self.std_data ** 2
        hyp = sq_sum.sqrt()
        c_skip = self.std_data ** 2 / sq_sum
        c_out = std * self.std_data / hyp
        c_in = 1 / hyp
        c_noise = th.log(std)

        nlf = self.noise_level_features(c_noise[:,:,0])
        model_out = model(c_in * x_t, nlf)
        pred_x0 = c_skip * x_t + c_out * model_out
        return pred_x0, self.proj_u(nlf)[:,0]

    def training_sample(self, model: Denoiser, x0: X) -> tuple[X,U,U]:
        """sample denoised predictions and per-batch loss weights"""
        b = x0.size(0)
        u = (th.arange(b) + th.rand(1)) / b # low discrepancy sampling
        t = self.std_noise(u)[:,None,None].to(x0.device)
        loss_weight = (t ** 2 + self.std_data ** 2) / (t * self.std_data) ** 2
        x_t = x0 + th.randn_like(x0) * t

        return *self.pred_x0(model, x_t, t), loss_weight[:,0,0]
    
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
        sigmas = th.tensor([*sigmas.tolist(), 0], device=z.device)
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
            x0_hat = self.pred_x0(denoiser, x, t)[0]
            score = (x0_hat - x) / t ** 2
            return -t * score

        x_t = z * (sigmas[0,0,0,0] ** 2 + self.std_data ** 2) ** .5
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
