# -*- coding: utf-8 -*-
import torch
from torch.optim.optimizer import Optimizer, required
import math
from copy import deepcopy


class ASGD(Optimizer):
    r"""Implements accelerated stochastic gradient descent. The federated version of this optimization
    scheme is proposed in 'Federated Accelerated Stochastic Gradient Descent'
    -> https://arxiv.org/abs/2006.08950

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr_eta (float): learning rate
        lr_gamma (float) : learning rate
        alpha : coupling coefficient
        beta : coupling coefficient
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    """

    def __init__(self, params, lr_eta=1e-3, local_step=10, lambd=0.001,
                 weight_decay=0, args=None):
        self.lr_eta = lr_eta
        self.lr_gamma = max(math.sqrt(lr_eta/(lambd*local_step)), lr_eta)
        self.alpha = 1/(self.lr_gamma*lambd)
        self.beta = self.alpha + 1
        defaults = dict(lr_eta=self.lr_eta, lr_gamma=self.lr_gamma,
                        alpha=self.alpha, beta=self.beta,
                        weight_decay=weight_decay, args=args)
        super(ASGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ASGD, self).__setstate__(state)

    def get_beta(self):
        return self.beta
    
    def store_current(self, **kargs):
        """Store current parameters as a buffer"""
        for group in self.param_groups:

            for p in group['params']:
                param_state = self.state[p]
                param_state['prev_model_buffer'] = p.data.clone().detach() 


    def step(self, closure=None, apply_lr=True, **kargs):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # retrieve para.
            weight_decay = group['weight_decay']
            lr_gamma = group['lr_gamma']
            alpha = group['alpha']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]

                # add weight decay.
                if weight_decay != 0 and apply_lr:
                    d_p.add_(p.data,alpha=weight_decay)

                # gradient step with coupling two parameters (update w_{k, t}^m in the FedAQ paper)
                p.data.mul_(1/alpha).add_(param_state['prev_model_buffer'], alpha=1 - 1/alpha).add_(d_p, alpha=-lr_gamma)
                
        return loss
