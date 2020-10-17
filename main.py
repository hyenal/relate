"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import matplotlib
import os
import time
import copy
import torch
from shutil import copyfile


# torch imports
import numpy as np

# arg loaders
import argparse
import json
from collections import namedtuple

# Import models
from models import archs

# Import datasets
from dataset import init_dataset

# Import optimizers
from optimizer import init_optimizer

from tools.utils import auto_init_args, get_net_input, get_visdom_env
from tools.stats import Stats
from tools.model_io import find_last_checkpoint, purge_epoch, \
                           load_model, get_checkpoint, save_model
matplotlib.use('Agg')


def init_model(cfg, force_load=False, clear_stats=False, add_log_vars=None):

    try:
        model = archs[cfg.default_opts['model']](**cfg.MODEL)
    except KeyError:
        raise KeyError('Unknown mode type: %s' % (cfg.default_opts['model']))

    # obtain the network outputs that should be logged
    if hasattr(model, 'log_vars'):
        log_vars = copy.deepcopy(model.log_vars)
    else:
        log_vars = ['objective']
    if add_log_vars is not None:
        log_vars.extend(copy.deepcopy(add_log_vars))

    visdom_env_charts = get_visdom_env(cfg) + "_charts"

    # init stats struct
    stats = Stats(log_vars, visdom_env=visdom_env_charts,
                  verbose=False, visdom_server=cfg.visdom_server,
                  visdom_port=cfg.visdom_port)

    if not cfg.path_to_last:
        cfg.path_to_last = cfg.exp_dir

    # find the last checkpoint
    if cfg.resume_epoch > 0:
        model_path = get_checkpoint(cfg.path_to_last, cfg.resume_epoch)
    else:
        model_path = find_last_checkpoint(cfg.path_to_last)

    optimizer_state = None

    if model_path is not None:
        print("found previous model %s" % model_path)
        if force_load or cfg.resume:
            print("   -> resuming")
            model_state_dict, stats_load, optimizer_state = load_model(
                model_path)

            if not cfg.clear_stats and stats_load is not None:
                stats = stats_load
            else:
                print("   -> clearing stats")

            if cfg.clear_optimizer:
                optimizer_state = None
                print("   -> clearing optimizer variables")

            model.load_state_dict(model_state_dict, strict=False)
            model.log_vars = log_vars
        else:
            print("   -> but not resuming -> starting from scratch")

    # update in case it got lost during load:
    stats.visdom_env = visdom_env_charts
    stats.visdom_server = cfg.visdom_server
    stats.visdom_port = cfg.visdom_port
    stats.plot_file = os.path.join(cfg.exp_dir, 'train_stats.pdf')
    stats.synchronize_logged_vars(log_vars)

    return model, stats, optimizer_state


def run(cfg):
    '''
    run the training loops
    '''

    # torch gpu setup
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_idx)

    # make the exp dir
    os.makedirs(cfg.exp_dir, exist_ok=True)

    # set the seeds
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # set cudnn to reproducibility mode
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("Loading dataset...")
    # setup datasets
    dset_train, dset_val = init_dataset(dataset=cfg.default_opts['dataset'],
                                        **cfg.DATASET, eval_only=cfg.eval_only)

    # init loaders
    trainloader = torch.utils.data.DataLoader(dset_train,
                                              num_workers=cfg.num_workers,
                                              pin_memory=True,
                                              batch_size=cfg.batch_size,
                                              shuffle=True)

    if dset_val is not None:
        valloader = torch.utils.data.DataLoader(dset_val,
                                                num_workers=cfg.num_workers,
                                                pin_memory=True,
                                                batch_size=cfg.batch_size,
                                                shuffle=False)
    else:
        valloader = None

    # test loaders
    eval_vars = None

    # init the model
    model, stats, optimizer_state = init_model(cfg, add_log_vars=eval_vars)
    start_epoch = stats.epoch + 1

    # move model to gpu
    if torch.cuda.is_available():
        model.cuda()

    optimizer, scheduler = init_optimizer(
        model, optimizer_state=optimizer_state, **cfg.SOLVER)

    print("Starting main loop...")
    # If evaluation just run it now and exit
    if cfg.eval_only:
        with stats:
            trainvalidate(cfg, model, stats, 0, valloader,
                          [namedtuple('dummyopt', 'num_iter')(num_iter=1)],
                          True, visdom_env_root=get_visdom_env(cfg),
                          exp_dir=cfg.exp_dir)
        return

    for epoch in range(start_epoch, cfg.SOLVER['max_epochs']):
        with stats:  # automatic new_epoch and plotting at every epoch start

            # train loop
            trainvalidate(cfg, model, stats, epoch, trainloader, optimizer,
                          False, visdom_env_root=get_visdom_env(cfg),
                          exp_dir=cfg.exp_dir)

            if valloader is not None:
                # val loop
                trainvalidate(cfg, model, stats, epoch, valloader,
                              [namedtuple('dummyopt', 'num_iter')(num_iter=1)],
                              True, visdom_env_root=get_visdom_env(cfg),
                              exp_dir=cfg.exp_dir)

            assert stats.epoch == epoch, "inconsistent stats!"

            # delete previous models if required
            if cfg.store_checkpoints_purge > 0:
                for prev_epoch in range(epoch-cfg.store_checkpoints_purge):
                    purge_epoch(cfg.exp_dir, prev_epoch)

            if cfg.store_checkpoints:
                outfile = get_checkpoint(cfg.exp_dir, epoch)
                save_model(model, stats, outfile, optimizer=optimizer)

            for sch in scheduler:
                sch.step()


def trainvalidate(cfg,
                  model,
                  stats,
                  epoch,
                  loader,
                  optimizer,
                  validation,
                  bp_var='objective',
                  visdom_env_root='trainvalidate',
                  exp_dir=''):

    if validation:
        model.eval()
        trainmode = 'val'
    else:
        model.train()
        trainmode = 'train'

    t_start = time.time()

    # clear the visualisations on the first run in the epoch
    clear_visualisations = True

    # get the visdom env name
    visdom_env_imgs = visdom_env_root + "_images_" + trainmode

    n_batches = len(loader)
    for it, batch in enumerate(loader):

        last_iter = it == n_batches-1

        # move to gpu where possible
        net_input = get_net_input(batch)

        # Optim will be perform
        for opt_num in range(len(optimizer)):
            for opt_iter in range(optimizer[opt_num].num_iter):

                # the forward pass
                if (not validation):
                    optimizer[opt_num].zero_grad()
                    preds, loss = model(trainmode='train', **net_input,
                                        it=it + n_batches * epoch,
                                        gen=(optimizer[opt_num].name == 'gen'))
                else:
                    with torch.no_grad():
                        preds, _ = model(trainmode='val', **net_input,
                                         it=it + n_batches * epoch,
                                         exp_dir=exp_dir,
                                         gen=True)

                # update the stats logger
                stats.update(preds, time_start=t_start,
                             stat_set=trainmode,
                             freeze_iter=not((opt_num == len(optimizer)-1) and
                                             (opt_iter == optimizer[opt_num].num_iter-1)))

                if opt_num == len(optimizer)-1 and\
                   opt_iter == optimizer[opt_num].num_iter-1:
                    # make sure we dont overwrite something
                    assert not any(k in preds.keys() for k in net_input.keys())
                    preds.update(net_input)  # merge everything into one dict

                    # print textual status update
                    if (it % cfg.metric_print_interval) == 0 or last_iter:
                        stats.print(stat_set=trainmode, max_it=n_batches)

                    # visualize results
                    if ((cfg.visualize_interval > 0) and (it % cfg.visualize_interval) == 0)\
                       or ((cfg.visualize_interval == 0) and (it == n_batches-2))\
                       or (validation and it == 0):
                        model.visualize(visdom_env_imgs, trainmode,
                                        preds, stats,
                                        clear_env=clear_visualisations,
                                        exp_dir=exp_dir,
                                        show_gt=(loader.dataset.__class__.__name__ != 'Dummy'))
                        clear_visualisations = False

                # optimizer step
                if (not validation):
                    loss.backward()
                    optimizer[opt_num].step()


class MainConfig(object):
    def __init__(self,
                 eval_only=False,
                 exp_dir='./relate/',
                 path_to_last='',
                 gpu_idx=0,
                 resume=True,
                 clear_stats=False,
                 clear_optimizer=False,
                 seed=0,
                 resume_epoch=-1,
                 store_checkpoints=True,
                 store_checkpoints_purge=1,
                 batch_size=100,
                 num_workers=4,
                 visdom_env='',
                 visdom_server='http://localhost',
                 visdom_port=8097,
                 metric_print_interval=30,
                 visualize_interval=0,
                 default_opts={'model': 'relate_static', 'dataset': 'clevr5',
                               'optimizer': 'adam'},
                 SOLVER={},
                 DATASET={},
                 MODEL={}
                 ):
        auto_init_args(self)


if __name__ == '__main__':
    # Get arguments
    parser = argparse.ArgumentParser('RELATE main arguments')
    parser.add_argument('--config_file', required=True, type=str)
    args = parser.parse_args()

    print("Loading config file...")
    # Load config file
    with open(args.config_file, 'r') as j:
        cfg_args = json.loads(j.read())

    # Build exp config now
    cfg = MainConfig(**cfg_args)

    # Dump config file
    os.makedirs(cfg.exp_dir, exist_ok=True)
    copyfile(args.config_file, cfg.exp_dir + '/config.json')

    run(cfg)