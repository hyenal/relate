"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch


def init_optimizer(model,
                   optimizer_state,
                   group_params={'default': {}},
                   opt_type='adam',
                   weight_decay=0.000,
                   lr_policy='multistep',
                   lr=0.0001,
                   gamma=0.1,
                   momentum=0.9,
                   betas=[0., 0.999],
                   milestones=(2000,),
                   max_epochs=300):
    optim, sched = [], []
    
    # init the optimizer
    if hasattr(model, '_get_param_groups') and model.custom_param_groups:
        # use the model function
        param_groups = model._get_param_groups()
    else:
        allprm = [prm for prm in model.parameters() if prm.requires_grad]
        param_groups = {'default': [{'params': allprm, 'lr': lr}]}

    # init the optimizer
    for opt_key, opt_params in group_params.items():
        try:
            p_groups = param_groups[opt_key]
        except KeyError:
            raise KeyError('Unknown param group %s'%(opt_key))

        if opt_params.get('type', opt_type) == 'sgd':
            optimizer = torch.optim.SGD(p_groups, lr=opt_params.get('lr', lr),
                                        momentum=opt_params.get('momentum',
                                                                momentum),
                                        weight_decay=weight_decay)

        elif opt_params.get('type', opt_type) == 'adagrad':
            optimizer = torch.optim.Adagrad(p_groups, lr=opt_params.get('lr',
                                                                        lr),
                                            weight_decay=weight_decay)

        elif opt_params.get('type', opt_type) == 'adam':
            optimizer = torch.optim.Adam(p_groups, lr=opt_params.get('lr', lr),
                                         betas=tuple(opt_params.get('betas', betas)),
                                         weight_decay=weight_decay)

        else:
            raise ValueError("no such solver type %s"
                             % opt_params.get('type', opt_type))

        print("  -> solver type = %s" % opt_params.get('type', opt_type))

        # Init scheduler
        if lr_policy == 'multistep':
            scheduler = torch.optim.lr_scheduler.\
                          MultiStepLR(optimizer, milestones=milestones,
                                      gamma=opt_params.get('gamma', gamma))
        else:
            raise ValueError("no such lr policy %s" % lr_policy)
        scheduler.max_epochs = opt_params.get('max_epochs', max_epochs)

        # give name to optimizer
        optimizer.name = opt_key
        optimizer.num_iter = opt_params.get('num_iter', 1)

        if optimizer_state is not None:
            print("  -> setting loaded optimizer state")
            optimizer.load_state_dict(optimizer_state[opt_key])

        optimizer.zero_grad()

        optim.append(optimizer)
        sched.append(scheduler)

    # Sort optim and sched this could be smarter than this
    optim, sched = map(list, zip(*sorted(zip(optim, sched),
                                 key=lambda x: x[0].name)))
    return optim, sched
