from tqdm import tqdm

import torch
import torch.nn.functional as F

from osu_dreamer.signal import MAP_SIGNAL_DIM as X_DIM

def extract(a, ts, x_shape):
    batch_size = ts.shape[0]
    out = a.gather(-1, ts.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(ts.device)
    
class BetaSchedule:
    def __init__(self, betas, net):
        self._net = net
        
        # define beta schedule
        self.betas = betas
        self.timesteps = len(betas)

        # define alphas 
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.rsqrt(self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        
        # Improved DDPM, Eqn. 10
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod) # beta_tilde
        assert (self.posterior_variance[1:] != 0).all(), self.posterior_variance[1:]
        
    def net(self, x,a,t,ts):
        """predict the mean of the noise added to `x` at timestep `ts`"""
        return self._net(x,a,t,ts)

    def q_sample(self, x: "N,X,L", ts: "N,", noise=None) -> "N,X,L":
        """sample q(x_t|x_0) using the nice property"""
        if noise is None:
            noise = torch.randn_like(x)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, ts, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, ts, x.shape
        )

        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
    
    def model_eps_var(self, x, a, t, ts):
        model_eps = self.net(x,a,t,ts)
        model_var = extract(self.posterior_variance, ts, x.shape)
        return model_eps, model_var
    
    def p_eps_mean_var(self, x, a, t, ts):
        """sample from p(x_{t-1} | x_t)"""
        model_eps, model_var = self.model_eps_var(x,a,t,ts)
            
        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        betas_t = extract(self.betas, ts, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, ts, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, ts, x.shape)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model_eps / sqrt_one_minus_alphas_cumprod_t)
        
        return model_eps, model_mean, model_var
        
    @torch.no_grad()
    def sample(self, a: "N,A,L", t: "N,T,L", x: "N,X,L" = None, *, ddim=False) -> "N,X,L":
        """sample p(x)"""
        
        b,_,l = a.size()
        
        if x is None:
            # start from pure noise (for each example in the batch)
            x = torch.randn((b,X_DIM,l), device=a.device)

        print()
        for i in tqdm(list(reversed(range(self.timesteps))), desc='sampling loop time step'):
            ts = torch.full((b,), i, device=a.device, dtype=torch.long)
            
            _, model_mean, model_var = self.p_eps_mean_var(x,a,t,ts)

            if i == 0 or ddim:
                x = model_mean
            else:
                x = model_mean + torch.sqrt(model_var) * torch.randn_like(x)
            
        return x
    
class CosineBetaSchedule(BetaSchedule):
    def __init__(self, timesteps, net, s=0.008):
        """
        cosine schedule as proposed in https://arxiv.org/abs/2102.09672
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)

        super().__init__(betas, net)

    
class StridedBetaSchedule(BetaSchedule):
    def __init__(self, schedule, steps, *args, **kwargs):
        # use_timesteps = set(torch.linspace(1, schedule.timesteps, steps).round().int().tolist())
        use_timesteps = set(torch.arange(1,schedule.timesteps, schedule.timesteps/steps).round().int().tolist())
        self.ts_map = []

        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(schedule.alphas_cumprod):
            if i in use_timesteps:
                self.ts_map.append(i)
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                
        super().__init__(torch.tensor(new_betas), *args, **kwargs)
                
            
    def net(self, x,a,t,ts):
        ts = torch.tensor(self.ts_map, device=ts.device, dtype=ts.dtype)[ts]
        return super().net(x,a,t,ts)