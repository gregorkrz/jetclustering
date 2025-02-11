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

from src.utils.train_utils import count_parameters, get_gt_func, get_loss_func
from src.utils.utils import clear_empty_paths
from src.utils.wandb_utils import get_run_by_name, update_args
from src.logger.logger import _logger, _configLogger
from src.dataset.dataset import SimpleIterDataset
from src.utils.import_tools import import_module
from src.utils.train_utils import (
    to_filelist,
    train_load,
    test_load,
    get_model,
    get_optimizer_and_scheduler,
    get_model_obj_score
)
from src.evaluation.clustering_metrics import compute_f1_score_from_result
from src.dataset.functions_graph import graph_batch_func
from src.utils.parser_args import parser
from src.utils.paths import get_path
import warnings
import pickle
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

if args.load_from_run:
    print("Loading args from run", args.load_from_run)
    run = get_run_by_name(args.load_from_run)
    args = update_args(args, run)
timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")
args.run_name = f"{args.run_name}_{timestamp}"
if args.load_model_weights:
    args.load_model_weights = get_path(args.load_model_weights, "results", fallback=True)
if args.load_objectness_score_weights:
    args.load_objectness_score_weights = get_path(args.load_objectness_score_weights, "results", fallback=True)
run_path = os.path.join(args.prefix, "train", args.run_name)
clear_empty_paths(get_path(os.path.join(args.prefix, "train"), "results"))  # Clear paths of failed runs that don't have any files or folders in them
run_path = get_path(run_path, "results")
#Path(run_path).mkdir(parents=True, exist_ok=False)
os.makedirs(run_path, exist_ok=False)
assert os.path.exists(run_path)
print("Created directory", run_path)
args.run_path = run_path
wandb.init(project=args.wandb_projectname, entity=os.environ["SVJ_WANDB_ENTITY"])
wandb.run.name = args.run_name
print("Setting the run name to", args.run_name)
#wandb.config.run_path = run_path
wandb.config.update(args.__dict__)
wandb.config.env_vars = {key: os.environ[key] for key in os.environ if key.startswith("SVJ_") or key.startswith("CUDA_") or key.startswith("SLURM_")}
if args.tag:
    wandb.run.tags = [args.tag.strip()]
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
from src.utils.nn.tools_condensation import train_epoch
from src.utils.nn.tools_condensation import evaluate as evaluate

training_mode = bool(args.data_train)
if training_mode:
    # val_loaders and test_loaders are a dictionary file -> dataloader with only one dataset
    # train_loader is a single dataloader of all the files
    train_loader, val_loaders, val_dataset = train_load(args)
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
        #if os.environ.get("CUDA_VISIBLE_DEVICES", None) is not None:
        #    gpus = [int(i) for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        dev = torch.device(gpus[0])
        local_rank = 0
else:
    gpus = None
    local_rank = 0
    dev = torch.device("cpu")

model = get_model(args, dev)

if args.train_objectness_score:
    model_obj_score = get_model_obj_score(args, dev)
    model_obj_score = model_obj_score.to(dev)
else:
    model_obj_score = None
num_parameters_counted = count_parameters(model)
print("Number of parameters:", num_parameters_counted)
wandb.config.num_parameters = num_parameters_counted

orig_model = model
loss = get_loss_func(args)
gt = get_gt_func(args)
batch_config = {"use_p_xyz": True, "use_four_momenta": False}

if "lgatr" in args.network_config.lower():
    batch_config = {"use_four_momenta": True}

batch_config["quark_dist_loss"] = args.loss == "quark_distance"
batch_config["obj_score"] = args.train_objectness_score
print("batch_config:", batch_config)
if training_mode:
    model = orig_model.to(dev)
    if args.backend is not None:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print("device_ids = gpus", gpus)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=gpus,
            output_device=local_rank,
            find_unused_parameters=True,
        )
    opt, scheduler = get_optimizer_and_scheduler(args, model, dev)
    if args.train_objectness_score:
        opt_os, scheduler_os = get_optimizer_and_scheduler(args, model_obj_score, dev, load_model_weights="load_objectness_score_weights")
    else:
        opt_os, scheduler_os = None, None
    # DataParallel
    if args.backend is None:
        if gpus is not None and len(gpus) > 1:
            # model becomes `torch.nn.DataParallel` w/ model.module being the original `torch.nn.Module`
            model = torch.nn.DataParallel(model, device_ids=gpus)
    if local_rank == 0:
        wandb.watch(model, log="all", log_freq=10)
    # Training loop
    best_valid_metric = np.inf
    grad_scaler = torch.cuda.amp.GradScaler() if args.use_amp else None
    steps = 0
    evaluate(
        model,
        val_loaders,
        dev,
        0,
        steps,
        loss_func=loss,
        gt_func=gt,
        local_rank=local_rank,
        args=args,
        batch_config=batch_config,
        predict=False,
        model_obj_score=model_obj_score
    )
    res = evaluate(
        model,
        val_loaders,
        dev,
        0,
        steps,
        loss_func=loss,
        gt_func=gt,
        local_rank=local_rank,
        args=args,
        batch_config=batch_config,
        predict=True,
        model_obj_score=model_obj_score
    )
    # It was the quickest to do it like this
    if model_obj_score is not None:
        res, res_obj_score_pred, res_obj_score_target = res
    f1 = compute_f1_score_from_result(res, val_dataset)
    wandb.log({"val_f1_score": f1}, step=steps)
    epochs = args.num_epochs
    if args.num_steps != -1:
        epochs = 999999999
    for epoch in range(1, epochs + 1):
        _logger.info("-" * 50)
        _logger.info("Epoch #%d training" % epoch)
        steps = train_epoch(
            args,
            model,
            loss_func=loss,
            gt_func=gt,
            opt=opt,
            scheduler=scheduler,
            train_loader=train_loader,
            dev=dev,
            epoch=epoch,
            grad_scaler=grad_scaler,
            local_rank=local_rank,
            current_step=steps,
            val_loader=val_loaders,
            batch_config=batch_config,
            val_dataset=val_dataset,
            obj_score_model=model_obj_score,
            opt_obj_score=opt_os,
            sched_obj_score=scheduler_os
        )
        if steps == "quit_training":
            break

if args.data_test:
    if args.backend is not None and local_rank != 0:
        sys.exit(0)
    if training_mode:
        del train_loader, val_loaders
        test_loaders = test_load(args)
    model = orig_model.to(dev)

    if gpus is not None and len(gpus) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpus)
    model = model.to(dev)
    i = 0
    for filename, test_loader in test_loaders.items():
        result = evaluate(
            model,
            test_loader,
            dev,
            0,
            0,
            loss_func=loss,
            gt_func=gt,
            local_rank=local_rank,
            args=args,
            batch_config=batch_config,
            predict=True,
            model_obj_score=model_obj_score
        )
        if model_obj_score is not None:
            result, result_obj_score, result_obj_score_target = result
            result["obj_score_pred"] = result_obj_score
            result["obj_score_target"] = result_obj_score_target
        _logger.info(f"Finished evaluating {filename}")
        result["filename"] = filename
        os.makedirs(run_path, exist_ok=True)
        output_filename = os.path.join(run_path, f"eval_{i}.pkl")
        pickle.dump(result, open(output_filename, "wb"))
        i += 1
