# -*- coding: utf-8 -*-
from fedtorch.components.optimizers.sgd import SGD
from fedtorch.components.optimizers.adam import AdamW
from fedtorch.components.optimizers.asgd import ASGD


def define_optimizer(args, model):
    # define the param to optimize.
    params_dict = dict(model.named_parameters())
    params = [
        {
            'params': [value],
            'name': key,
            'weight_decay': args.weight_decay if 'bn' not in key else 0.0
        }
        for key, value in params_dict.items()
    ]

    # define the optimizer.
    if args.optimizer == 'sgd':
        return SGD(
            params, lr=args.learning_rate,
            in_momentum=args.in_momentum_factor,
            out_momentum=(args.out_momentum_factor
                          if args.out_momentum_factor is not None
                          else 1.0 - 1.0 / args.graph.n_nodes),
            nesterov=args.use_nesterov, args=args)
    elif args.optimizer == 'asgd':
        return ASGD(
            params, lr_eta=args.learning_rate,
            local_step=args.local_step, lambd=args.lambd,
            args=args
        )
    else:
        return AdamW(
            params, lr=args.learning_rate,
            correct_wd=args.correct_wd
        )
