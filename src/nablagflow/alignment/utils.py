import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_size_list(total, bs):
    return [bs] * (total // bs) + ([total % bs] if total % bs != 0 else [])

def normal_logp(x, mean=0., sigma=1., mean_over_dim=False):
    # x: (bs, ...)
    # mean: scalar or (bs, ...)
    # sigma: scalar float or (bs, 1); assume all dimensions have the same sigma

    if isinstance(mean, torch.Tensor):
        assert mean.ndim == x.ndim
    if isinstance(sigma, torch.Tensor):
        assert sigma.ndim == 2
        log_sigma = torch.log(sigma)
        # make sigma to the same dimension as x
        sigma = sigma[None].expand_as(x)
        log_sigma = log_sigma[None].expand_as(x)
    else:
        log_sigma = np.log(sigma)

    neg_logp = 0.5 * np.log(2 * np.pi) + log_sigma \
               + 0.5 * (x - mean) ** 2 / (sigma ** 2)

    if mean_over_dim:
        return torch.mean(-neg_logp, dim=list(range(1, x.ndim))) # (bs,)
    else:
        return torch.sum(-neg_logp, dim=list(range(1, x.ndim))) # (bs,)

class TrajectoryBuffer:
    def __init__(self, buffer_size):
        self.buffer = {}
        self.count = 0
        self.real_size = 0
        self.size = buffer_size

    def add(self, dic):
        for k, v in dic.items():
            if k not in self.buffer:
                self.buffer[k] = v
            else:
                self.buffer[k] = torch.cat([self.buffer[k], v], dim=0)
            real_size = self.buffer[k].size(0)
            if real_size > self.size:
                self.buffer[k] = self.buffer[k][real_size - self.size:]

        self.real_size = min(self.real_size + dic[k].size(0), self.size)
        # self.count = (self.count + 1) % self.size

    def sample(self, batch_size):
        idx = np.random.choice(self.real_size, batch_size, replace=False)
        return {k: v[idx] for k, v in self.buffer.items()}


def image_postprocess(x, no_clamp=False):
    # [-1, 1] -> [0, 1]
    if no_clamp:
        return (x + 1) / 2
    return torch.clamp((x + 1) / 2, 0, 1)  # x / 2 + 0.5

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

from contextlib import contextmanager
@contextmanager
def freeze(module: torch.nn.Module):
    """
    Disable gradient for all module parameters. However, if input requires grad
    the graph will still be constructed.
    """
    try:
        prev_states = [p.requires_grad for p in module.parameters()]
        for p in module.parameters():
            p.requires_grad_(False)
        yield

    finally:
        for p, state in zip(module.parameters(), prev_states):
            p.requires_grad_(state)

