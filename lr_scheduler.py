import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler


import math
from torch.optim.lr_scheduler import _LRScheduler

class LambdaWarmUpCosineScheduler(_LRScheduler):
    def __init__(self, 
                 optimizer,
                 warmup_steps,
                 lr_min,
                 lr_max,
                 lr_start,
                 max_decay_steps):
        self.optimizer= optimizer
        self.lr_warmup_steps = warmup_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.
        self.last_step = 0.
        super(LambdaWarmUpCosineScheduler, self).__init__(optimizer)
        
    def get_lr(self):
        if self.last_step < self.lr_warmup_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warmup_steps * self.last_step + self.lr_start
            self.last_lr = lr
        else:
            t = (self.last_step - self.lr_warmup_steps) / (self.lr_max_decay_steps - self.lr_warmup_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                    1 + np.cos(t * np.pi))
            self.last_lr = lr
        return [lr for _ in self.optimizer.param_groups]
    
    def step(self, step=None):
        if step is None:
            self.last_step += 1
        else:
            self.last_step = step
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr