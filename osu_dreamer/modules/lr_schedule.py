
from dataclasses import dataclass

@dataclass(kw_only=True)
class LRScheduleArgs:
    warmup_steps: int = 0
    warmup_init: float = 1
    decay_start: float

def make_lr_schedule(lr: LRScheduleArgs):
    assert lr.warmup_steps <= lr.decay_start

    def schedule(step: int) -> float:
        if step < lr.warmup_steps:
            # exponential warmup
            return lr.warmup_init ** (1 - step / lr.warmup_steps)
        elif step > lr.decay_start:
            # isqrt decay
            return (step / lr.decay_start) ** -.5
        return 1.
    
    return schedule