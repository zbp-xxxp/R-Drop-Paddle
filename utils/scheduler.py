import math
from paddle.optimizer.lr import LRScheduler

class WarmupConstantSchedule(LRScheduler):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, warmup_steps, learning_rate, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.steps = -1
        super(WarmupConstantSchedule, self).__init__(learning_rate, last_epoch, verbose=True)

    def get_lr(self):
        self.steps += 1
        if self.steps < self.warmup_steps:
            return float(self.steps) / float(max(1.0, self.warmup_steps))
        return 1.


class WarmupLinearSchedule(LRScheduler):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, learning_rate, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.steps = -1
        super(WarmupLinearSchedule, self).__init__(learning_rate, last_epoch, verbose=True)

    def get_lr(self):
        self.steps += 1
        if self.steps < self.warmup_steps:
            return float(self.steps) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - self.steps) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LRScheduler):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, learning_rate, warmup_steps, t_total, last_epoch=-1, cycles=.5):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        self.steps = -1
        super(WarmupCosineSchedule, self).__init__(learning_rate, last_epoch, verbose=True)

    def get_lr(self):
        self.steps += 1
        if self.steps < self.warmup_steps:
            return float(self.steps) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(self.steps - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
