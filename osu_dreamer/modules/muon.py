
import torch

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

def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    orig_shape = update.shape
    if update.ndim > 2:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(0) / update.size(1))**0.5
    return update.view(orig_shape)

def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class Muon(torch.optim.Optimizer):
    def __init__(
        self, 
        param_groups,
        lr=.02,
        adam_lr=3e-4,
        weight_decay=0.1,
        momentum=0.95, 
        adamw_betas=(0.9, 0.95), 
        adamw_eps=1e-10,
    ):
        super().__init__(param_groups, dict(
            lr=lr,
            adam_lr=adam_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            adamw_betas=adamw_betas, 
            adamw_eps=adamw_eps,
        ))

    @torch.no_grad
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if p.grad is None:
                    continue

                if p.ndim < 2:
                    # adamw update
                    lr = group["adam_lr"]
                    if 'step' not in state:
                        state['step'] = 0
                        state['moment1'] = torch.zeros_like(p.grad)
                        state['moment2'] = torch.zeros_like(p.grad)
                    state['step'] += 1
                    update = adam_update(
                        p.grad, 
                        state['moment1'], state['moment2'], state['step'], 
                        group['adamw_betas'], group['adamw_eps'],
                    )
                else:
                    # muon update
                    lr = group["lr"]
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(p.grad)
                    update = muon_update(p.grad, state["momentum_buffer"], group["momentum"])

                p.mul_(1 - lr * group['weight_decay'])
                p.add_(update, alpha=-lr)