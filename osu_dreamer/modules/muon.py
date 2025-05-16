
import torch
from typing import Generator

# @torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps) # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    - https://github.com/KellerJordan/Muon/blob/master/muon.py
    - https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py
    """
    def __init__(
        self, 
        muon_params, 
        lr=1e-3, 
        weight_decay=0.1,
        momentum=0.95, 
        nesterov=True, 
        ns_steps=6,
        adamw_params=None, 
        adamw_betas=(0.95, 0.95), 
        adamw_eps=1e-8, 
    ):

        defaults = dict(
            lr=lr, 
            momentum=momentum, 
            nesterov=nesterov, 
            weight_decay=weight_decay,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas, 
            adamw_eps=adamw_eps, 
        )

        # handle list of params or list of dicts
        if isinstance(muon_params, Generator):
            muon_params = list(muon_params)
        if isinstance(adamw_params, Generator):
            adamw_params = list(adamw_params)
        elif adamw_params is None:
            adamw_params = []

        super().__init__([*muon_params, *adamw_params], defaults)

        # Sort parameters into those for which we will use Muon, and those for which we will not
        # we cant pickle booleans for saving, so we will use 1=True, 0=False
        def assign_muon(p):
            if p.ndim >= 2 and p.size(0) < 10000:
                self.state[p]['use_muon'] = 1
            else:
                self.state[p]['use_muon'] = 0

        if isinstance(muon_params[0], dict):
            for group in muon_params:
                for p in group['params']:
                    assign_muon(p)
        else:
            for p in muon_params:
                assign_muon(p)

        def assign_adamw(p):
            # Do not use Muon for parameters in adamw_params
            self.state[p]['use_muon'] = 0

        if len(adamw_params) and isinstance(adamw_params[0], dict):
            for group in adamw_params:
                for p in group['params']:
                    assign_adamw(p)
        else:
            for p in adamw_params:
                assign_adamw(p)

        if torch.distributed.is_initialized():
            self.world_size = torch.distributed.get_world_size()
            self.rank = torch.distributed.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            for p in group['params']:
                state = self.state[p]
                g = p.grad
                if g is None:
                    continue

                if state['use_muon'] == 1:
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)

                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(g)
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(g)
                    if group['nesterov']:
                        g = g.add(buf, alpha=momentum)

                    # gives NaNs when done with Dtensor, instead of throwing a typical op not supported error, quite sneaky
                    g = zeropower_via_newtonschulz5(g, steps=group['ns_steps'])
                    p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(
                        g.view_as(p.data).type_as(p.data), 
                        alpha = -lr * .2 * max(g.size(0),g.size(1))**0.5,
                    )
                else:
                    if 'step' not in state:
                        state['step'] = 0
                        state['moment1'] = torch.zeros_like(g)
                        state['moment2'] = torch.zeros_like(g)
                    state['step'] += 1
                    step = state['step']
                    buf1 = state['moment1']
                    buf2 = state['moment2']
                    buf1.lerp_(g, 1-group['adamw_betas'][0])
                    buf2.lerp_(g.square(), 1-group['adamw_betas'][1])

                    g = buf1 / (group['adamw_eps'] + buf2.sqrt())

                    bias_correction1 = 1 - group['adamw_betas'][0]**step
                    bias_correction2 = 1 - group['adamw_betas'][1]**step
                    scale = bias_correction1 / bias_correction2**0.5
                    p.data.mul_(1 - lr * weight_decay)
                    p.data.add_(g, alpha=-lr/scale)


