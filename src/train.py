#!/usr/bin/env python

import shutil
import glob
import argparse
import functools
import numpy as np
import math
import torch
import sys
import os
import wandb
import time
from pathlib import Path

torch.autograd.set_detect_anomaly(True)

from src.logger.logger import _logger, _configLogger
from src.dataset.dataset import SimpleIterDataset
from src.utils.import_tools import import_module
from src.utils.train_utils import (
    to_filelist,
    train_load,
    onnx,
    test_load,
    iotest,
    get_model,
    profile,
    optim,
    save_root,
    save_parquet,
)
from src.dataset.functions_graph import graph_batch_func
from src.utils.parser_args import parser
from src.utils.paths import get_path
import warnings

import os

def find_free_port():
    """https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number"""
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

# Create directories and initialize wandb run
args = parser.parse_args()
timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
args.run_name = f"{args.run_name}_{timestamp}"
if args.load_model_weights:
    args.load_model_weights = get_path(args.load_model_weights, "results")
run_path = os.path.join(args.prefix, "train", args.run_name)
run_path = get_path(run_path, "results")
Path(run_path).mkdir(parents=True, exist_ok=False)
wandb.init(project=args.wandb_projectname, entity=os.environ["SVJ_WANDB_ENTITY"])
wandb.run.name = args.wandb_displayname
wandb.config.run_path = run_path
wandb.config.update(args.__dict__)
wandb.config.env_vars = {key: os.environ[key] for key in os.environ if key.startswith("SVJ_")}

args.local_rank = (
    None if args.backend is None else int(os.environ.get("LOCAL_RANK", "0"))
)
if args.backend is not None:
    port = find_free_port()
    args.port = port
    world_size = torch.cuda.device_count()
stdout = sys.stdout
if args.local_rank is not None:
    args.log += ".%03d" % args.local_rank
    if args.local_rank != 0:
        stdout = None
_configLogger("weaver", stdout=stdout, filename=args.log)

warnings.filterwarnings("ignore")
from src.utils.nn.tools_condensation import train_regression as train
from src.utils.nn.tools_condensation import evaluate_regression as evaluate

training_mode = not args.predict
if training_mode:
    # val_loaders and test_loaders are a dictionary file -> dataloader with only one dataset
    # train_loader is a single dataloader of all the files
    train_loader, val_loaders = train_load(args)
else:
    test_loaders = test_load(args)


if args.gpus:
    if args.backend is not None:
        # distributed training
        local_rank = args.local_rank
        print("localrank", local_rank)
        torch.cuda.set_device(local_rank)
        gpus = [local_rank]
        dev = torch.device(local_rank)
        print("initializing group process", dev)
        torch.distributed.init_process_group(backend=args.backend)
        _logger.info(f"Using distributed PyTorch with {args.backend} backend")
        print("ended initializing group process")
    else:
        gpus = [int(i) for i in args.gpus.split(",")]
        dev = torch.device(gpus[0])
        local_rank = 0
else:
    gpus = None
    local_rank = 0
    dev = torch.device("cpu")

model, model_info, loss_func = get_model(args)
from src.utils.train_utils import count_parameters
num_parameters_counted = count_parameters(model)
print("Number of parameters:", num_parameters_counted)

orig_model = model
training_mode = not args.predict
if training_mode:
    model = orig_model.to(dev)
    if args.load_model_weights:
        model_path = args.load_model_weights
        _logger.info("Loading model %s for training from there on" % model_path)
        model.load_state_dict(torch.load(model_path, map_location=dev))
    print("MODEL DEVICE", next(model.parameters()).is_cuda)

    if args.backend is not None:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print("device_ids = gpus", gpus)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=gpus,
            output_device=local_rank,
            find_unused_parameters=True,
        )

    opt, scheduler = optim(args, model, dev)

    # DataParallel
    if args.backend is None:
        if gpus is not None and len(gpus) > 1:
            # model becomes `torch.nn.DataParallel` w/ model.module being the original `torch.nn.Module`
            model = torch.nn.DataParallel(model, device_ids=gpus)
    if args.log_wandb and local_rank == 0:
        wandb.watch(model, log="all", log_freq=10)
        # model = model.to(dev)

    # training loop
    best_valid_metric = np.inf
    grad_scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    steps = 0
    for epoch in range(args.num_epochs):
        if args.load_epoch is not None:
            if epoch <= args.load_epoch:
                continue
        _logger.info("-" * 50)
        _logger.info("Epoch #%d training" % epoch)
        if args.clustering_and_energy_loss and epoch > args.energy_loss_delay:
            print("Switching on energy loss!")
            add_energy_loss = True
        steps += train(
            model,
            opt,
            scheduler,
            train_loader,
            dev,
            epoch,
            steps_per_epoch=args.steps_per_epoch,
            current_step=steps,
            grad_scaler=grad_scaler,
            local_rank=local_rank,
            args=args,
        )

        if args.model_prefix and (args.backend is None or local_rank == 0):
            dirname = os.path.dirname(args.model_prefix)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)

            state_dict = (
                model.module.state_dict()
                if isinstance(
                    model,
                    (
                        torch.nn.DataParallel,
                        torch.nn.parallel.DistributedDataParallel,
                    ),
                )
                else model.state_dict()
            )

            torch.save(state_dict, args.model_prefix + "_epoch-%d_state.pt" % epoch)
            torch.save(
                opt.state_dict(),
                args.model_prefix + "_epoch-%d_optimizer.pt" % epoch,
            )
        # if args.backend is not None and local_rank == 0:
        # TODO: save checkpoint
        #     save_checkpoint()

        _logger.info("Epoch #%d validating" % epoch)
        valid_metric = evaluate(
            model,
            val_loader,
            dev,
            epoch,
            steps_per_epoch=args.steps_per_epoch_val,
            local_rank=local_rank,
            args=args,
        )
        is_best_epoch = valid_metric < best_valid_metric
        if is_best_epoch:
            print("Best epoch!")
            best_valid_metric = valid_metric
            if args.model_prefix and (args.backend is None or local_rank == 0):
                shutil.copy2(
                    args.model_prefix + "_epoch-%d_state.pt" % epoch,
                    args.model_prefix + "_best_epoch_state.pt",
                )
                # torch.save(model, args.model_prefix + '_best_epoch_full.pt')
        _logger.info(
            "Epoch #%d: Current validation metric: %.5f (best: %.5f)"
            % (epoch, valid_metric, best_valid_metric),
            color="bold",
        )

if args.data_test:
    tb = None
    if args.backend is not None and local_rank != 0:
        sys.exit(0)
    if args.log_wandb and local_rank == 0:
        import wandb
        from src.utils.logger_wandb import log_wandb_init

        wandb.init(project=args.wandb_projectname, entity=args.wandb_entity)
        wandb.run.name = args.wandb_displayname
        log_wandb_init(args, data_config)

    if training_mode:
        del train_loader, val_loader
        test_loaders, data_config = test_load(args)

    if not args.model_prefix.endswith(".onnx"):
        if args.predict_gpus:
            gpus = [int(i) for i in args.predict_gpus.split(",")]
            dev = torch.device(gpus[0])
        else:
            gpus = None
            dev = torch.device("cpu")
        model = orig_model.to(dev)
        if args.model_prefix:
            model_path = (
                args.model_prefix
                if args.model_prefix.endswith(".pt")
                else args.model_prefix + "_best_epoch_state.pt"
            )
            _logger.info("Loading model %s for eval" % model_path)
            model.load_state_dict(torch.load(model_path, map_location=dev))
        if gpus is not None and len(gpus) > 1:
            model = torch.nn.DataParallel(model, device_ids=gpus)
        model = model.to(dev)

    for name, get_test_loader in test_loaders.items():
        test_loader = get_test_loader()
        test_metric, scores, labels, observers = evaluate(
            model,
            test_loader,
            dev,
            epoch=None,
            for_training=False,
            loss_func=loss_func,
            steps_per_epoch=args.steps_per_epoch_val,
            tb_helper=tb,
            logwandb=args.log_wandb,
            energy_weighted=args.energy_loss,
            local_rank=local_rank,
            loss_terms=[args.clustering_loss_only, args.clustering_and_energy_loss],
            args=args,
        )

        _logger.info("Test metric %.5f" % test_metric, color="bold")
        del test_loader
