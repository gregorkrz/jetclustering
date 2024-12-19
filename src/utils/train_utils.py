import os
import ast
import glob
import functools
import math
import torch

from torch.utils.data import DataLoader
from src.logger.logger import _logger, _configLogger
from src.dataset.dataset import EventDatasetCollection, EventDataset
from src.utils.import_tools import import_module
from src.dataset.functions_graph import graph_batch_func
from src.dataset.functions_data import concat_events
from src.utils.paths import get_path

def to_filelist(args, mode="train"):
    if mode == "train":
        flist = args.data_train
    elif mode == "val":
        flist = args.data_val
    else:
        raise NotImplementedError("Invalid mode %s" % mode)
    print(mode, "filelist:", flist)
    flist = [get_path(p, "preprocessed_data") for p in flist]
    return flist

def train_load(args):
    train_files = to_filelist(args, "train")
    val_files = to_filelist(args, "val")
    train_data = EventDatasetCollection(train_files)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=min(args.num_workers, int(len(train_files) * args.file_fraction)),
        collate_fn=concat_events,
        persistent_workers=args.num_workers > 0 and args.steps_per_epoch is not None,
    )
    val_loaders = {}
    for filename in val_files:
        val_data = EventDataset.from_directory(filename, mmap=True)
        val_loaders[filename] = DataLoader(
            val_data,
            batch_size=args.batch_size,
            drop_last=True,
            pin_memory=True,
            collate_fn=concat_events,
            num_workers=min(args.num_workers, int(len(val_files) * args.file_fraction)),
            persistent_workers=args.num_workers > 0 and args.steps_per_epoch_val is not None,
        )
    return train_loader, val_loaders

def test_load(args):
    test_files = to_filelist(args, "test")
    test_loaders = {}
    for filename in test_files:
        test_data = EventDataset.from_directory(filename, mmap=True)
        test_loaders[filename] = DataLoader(
            test_data,
            batch_size=args.batch_size,
            drop_last=True,
            pin_memory=True,
            collate_fn=concat_events,
            num_workers=min(args.num_workers, int(len(test_files) * args.file_fraction)),
            persistent_workers=args.num_workers > 0 and args.steps_per_epoch_val is not None,
        )
    return test_loaders

def optim(args, model, device):
    """
    Optimizer and scheduler.
    :param args:
    :param model:
    :return:
    """
    optimizer_options = {k: ast.literal_eval(v) for k, v in args.optimizer_option}
    _logger.info("Optimizer options: %s" % str(optimizer_options))

    names_lr_mult = []
    if "weight_decay" in optimizer_options or "lr_mult" in optimizer_options:
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/optim/optim_factory.py#L31
        import re

        decay, no_decay = {}, {}
        names_no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if (
                len(param.shape) == 1
                or name.endswith(".bias")
                or (
                    hasattr(model, "no_weight_decay")
                    and name in model.no_weight_decay()
                )
            ):
                no_decay[name] = param
                names_no_decay.append(name)
            else:
                decay[name] = param

        decay_1x, no_decay_1x = [], []
        decay_mult, no_decay_mult = [], []
        mult_factor = 1
        if "lr_mult" in optimizer_options:
            pattern, mult_factor = optimizer_options.pop("lr_mult")
            for name, param in decay.items():
                if re.match(pattern, name):
                    decay_mult.append(param)
                    names_lr_mult.append(name)
                else:
                    decay_1x.append(param)
            for name, param in no_decay.items():
                if re.match(pattern, name):
                    no_decay_mult.append(param)
                    names_lr_mult.append(name)
                else:
                    no_decay_1x.append(param)
            assert len(decay_1x) + len(decay_mult) == len(decay)
            assert len(no_decay_1x) + len(no_decay_mult) == len(no_decay)
        else:
            decay_1x, no_decay_1x = list(decay.values()), list(no_decay.values())
        wd = optimizer_options.pop("weight_decay", 0.0)
        parameters = [
            {"params": no_decay_1x, "weight_decay": 0.0},
            {"params": decay_1x, "weight_decay": wd},
            {
                "params": no_decay_mult,
                "weight_decay": 0.0,
                "lr": args.start_lr * mult_factor,
            },
            {
                "params": decay_mult,
                "weight_decay": wd,
                "lr": args.start_lr * mult_factor,
            },
        ]
        _logger.info(
            "Parameters excluded from weight decay:\n - %s",
            "\n - ".join(names_no_decay),
        )
        if len(names_lr_mult):
            _logger.info(
                "Parameters with lr multiplied by %s:\n - %s",
                mult_factor,
                "\n - ".join(names_lr_mult),
            )
    else:
        parameters = model.parameters()

    if args.optimizer == "ranger":
        from src.utils.nn.optimizer.ranger import Ranger

        opt = Ranger(parameters, lr=args.start_lr, **optimizer_options)
    elif args.optimizer == "adam":
        opt = torch.optim.Adam(parameters, lr=args.start_lr, **optimizer_options)
    elif args.optimizer == "adamW":
        opt = torch.optim.AdamW(parameters, lr=args.start_lr, **optimizer_options)
    elif args.optimizer == "radam":
        opt = torch.optim.RAdam(parameters, lr=args.start_lr, **optimizer_options)

    # load previous training and resume if `--load-epoch` is set
    if args.load_epoch is not None:
        _logger.info("Resume training from epoch %d" % args.load_epoch)
        model_state = torch.load(
            args.model_prefix + "_epoch-%d_state.pt" % args.load_epoch,
            map_location=device,
        )
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(model_state)
        else:
            model.load_state_dict(model_state)
        opt_state_file = args.model_prefix + "_epoch-%d_optimizer.pt" % args.load_epoch
        if os.path.exists(opt_state_file):
            opt_state = torch.load(opt_state_file, map_location=device)
            opt.load_state_dict(opt_state)
        else:
            _logger.warning("Optimizer state file %s NOT found!" % opt_state_file)

    scheduler = None
    if args.lr_finder is None:
        if args.lr_scheduler == "steps":
            lr_step = round(args.num_epochs / 3)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                opt,
                milestones=[10],  # [lr_step, 2 * lr_step],
                gamma=0.20,
                last_epoch=-1 if args.load_epoch is None else args.load_epoch,
            )
        elif args.lr_scheduler == "flat+decay":
            num_decay_epochs = max(1, int(args.num_epochs * 0.3))
            milestones = list(
                range(args.num_epochs - num_decay_epochs, args.num_epochs)
            )
            gamma = 0.01 ** (1.0 / num_decay_epochs)
            if len(names_lr_mult):

                def get_lr(epoch):
                    return gamma ** max(0, epoch - milestones[0] + 1)  # noqa

                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    opt,
                    (lambda _: 1, lambda _: 1, get_lr, get_lr),
                    last_epoch=-1 if args.load_epoch is None else args.load_epoch,
                    verbose=True,
                )
            else:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    opt,
                    milestones=milestones,
                    gamma=gamma,
                    last_epoch=-1 if args.load_epoch is None else args.load_epoch,
                )
        elif args.lr_scheduler == "flat+linear" or args.lr_scheduler == "flat+cos":
            total_steps = args.num_epochs * args.steps_per_epoch
            warmup_steps = args.warmup_steps
            flat_steps = total_steps * 0.7 - 1
            min_factor = 0.001

            def lr_fn(step_num):
                if step_num > total_steps:
                    raise ValueError(
                        "Tried to step {} times. The specified number of total steps is {}".format(
                            step_num + 1, total_steps
                        )
                    )
                if step_num < warmup_steps:
                    return 1.0 * step_num / warmup_steps
                if step_num <= flat_steps:
                    return 1.0
                pct = (step_num - flat_steps) / (total_steps - flat_steps)
                if args.lr_scheduler == "flat+linear":
                    return max(min_factor, 1 - pct)
                else:
                    return max(min_factor, 0.5 * (math.cos(math.pi * pct) + 1))

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                opt,
                lr_fn,
                last_epoch=-1
                if args.load_epoch is None
                else args.load_epoch * args.steps_per_epoch,
            )
            scheduler._update_per_step = (
                True  # mark it to update the lr every step, instead of every epoch
            )
        elif args.lr_scheduler == "one-cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=args.start_lr,
                epochs=args.num_epochs,
                steps_per_epoch=args.steps_per_epoch,
                pct_start=0.3,
                anneal_strategy="cos",
                div_factor=25.0,
                last_epoch=-1 if args.load_epoch is None else args.load_epoch,
            )
            scheduler._update_per_step = (
                True  # mark it to update the lr every step, instead of every epoch
            )
        elif args.lr_scheduler == "reduceplateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, patience=2, threshold=0.01
            )
            # scheduler._update_per_step = (
            #     True  # mark it to update the lr every step, instead of every epoch
            # )
            scheduler._update_per_step = (
                False  # mark it to update the lr every step, instead of every epoch
            )
    return opt, scheduler


def get_model(args):
    """
    Loads the model
    :param args:
    :param data_config:
    :return: model, model_info, network_module, network_options
    """
    network_module = import_module(args.network_config, name="_network_module")
    network_options = {k: ast.literal_eval(v) for k, v in args.network_option}
    if args.export_onnx:
        network_options["for_inference"] = True
    if args.use_amp:
        network_options["use_amp"] = True
    if args.clustering_loss_only:
        network_options["output_dim"] = args.clustering_space_dim + 1
    else:
        network_options["output_dim"] = args.clustering_space_dim + 28
    network_options["input_dim"] = 9 + args.n_noise
    network_options.update(data_config.custom_model_kwargs)
    if args.use_heads:
        network_options["separate_heads"] = True
    _logger.info("Network options: %s" % str(network_options))
    if args.gpus:
        gpus = [int(i) for i in args.gpus.split(",")]  # ?
        dev = torch.device(gpus[0])
    else:
        gpus = None
        local_rank = 0
        dev = torch.device("cpu")
    model, model_info = network_module.get_model(
        data_config, args=args, dev=dev, **network_options
    )

    if args.freeze_core:
        model.mod.freeze("core")
        print("Frozen core parameters")
    if args.freeze_beta:
        model.mod.freeze("beta")
        print("Frozen beta parameters")
        assert model.mod.beta_weight == 1.0
        model.mod.beta_weight = 0.0
    if args.beta_zeros:
        model.mod.beta_exp_weight = 1.0
        print("Set beta_exp_weight to 1.0")
    if args.freeze_coords:
        model.mod.freeze("coords")
        print("Frozen coordinates parameters")
    if args.load_model_weights:
        print("Loading model state from %s" % args.load_model_weights)
        model_state = torch.load(args.load_model_weights, map_location="cpu")
        model_dict = model.state_dict()
        model_state = {k: v for k, v in model_state.items() if k in model_dict}
        model_dict.update(model_state)
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        _logger.info(
            "Model initialized with weights from %s\n ... Missing: %s\n ... Unexpected: %s"
            % (args.load_model_weights, missing_keys, unexpected_keys)
        )
        if args.copy_core_for_beta:
            model.mod.create_separate_beta_core()
            print("Created separate beta core")
    return model

def get_loss_func(args):
    # Loss function  takes in the output of a model and the output of GT (the GT labels) and returns the loss.

    def loss(model_output, gt_labels):


def get_gt_func(args):
    # Gets the GT function: the function accepts an Event batch
    # and returns the ground truth labels (GT idx of a dark quark it belongs to, or -1 for noise)
    # By default, it returns the dark quark that is closest to the event, IF it's closer than R.
    R = 0.8
    def get_idx_for_event(obj, i):
        return obj.batch_number[i], obj.batch_number[i + 1]
    def get_labels(b, pfcands):
        # b: Batch of events
        labels = torch.zeros(len(pfcands)).long()
        for i in range(len(b)):
            s, e = get_idx_for_event(b.matrix_element_gen_particles, i)
            dq_eta = b.matrix_element_gen_particles.eta[s:e]
            dq_phi = b.matrix_element_gen_particles.phi[s:e]
            # dq_pt = b.matrix_element_gen_particles.pt[s:e] # Maybe we can somehow weigh the loss by pt?
            s, e = get_idx_for_event(pfcands, i)
            pfcands_eta = pfcands.eta[s:e]
            pfcands_phi = pfcands.phi[s:e]
            # calculate the distance matrix between each dark quark and pfcands
            dist_matrix = torch.cdist(
                torch.stack([dq_eta, dq_phi], dim=1),
                torch.stack([pfcands_eta, pfcands_phi], dim=1),
                p=2
            )
            dist_matrix = dist_matrix.T
            closest_quark_dist, closest_quark_idx = dist_matrix.min(dim=1)
            closest_quark_idx[closest_quark_dist > R] = -1
            labels[s:e] = closest_quark_idx
        return labels
    def gt(events):
        #return {
        #    "pfcands": get_labels(events, events.pfcands),
        #    "special_pfcands": get_labels(events, events.special_pfcands)
        #}
        return torch.cat([get_labels(events, events.pfcands), get_labels(events, events.special_pfcands)])
    return gt


def iotest(args, data_loader):
    """
    Io test
    :param args:
    :param data_loader:
    :return:
    """
    from tqdm.auto import tqdm
    from collections import defaultdict
    from src.data.tools import _concat

    _logger.info("Start running IO test")
    monitor_info = defaultdict(list)

    for X, y, Z in tqdm(data_loader):
        for k, v in Z.items():
            monitor_info[k].append(v.cpu().numpy())
    monitor_info = {k: _concat(v) for k, v in monitor_info.items()}
    if monitor_info:
        monitor_output_path = "weaver_monitor_info.pkl"
        import pickle

        with open(monitor_output_path, "wb") as f:
            pickle.dump(monitor_info, f)
        _logger.info("Monitor info written to %s" % monitor_output_path)




def save_parquet(args, output_path, scores, labels, observers):
    """
    Saves as parquet file
    :param scores:
    :param labels:
    :param observers:
    :return:
    """
    import awkward as ak

    output = {"scores": scores}
    output.update(labels)
    output.update(observers)
    ak.to_parquet(ak.Array(output), output_path, compression="LZ4", compression_level=4)


def count_parameters(model):
    return sum(p.numel() for p in model.mod.parameters() if p.requires_grad)


def model_setup1(args, data_config):
    """
    Loads the model
    :param args:
    :param data_config:
    :return: model, model_info, network_module, network_options
    """
    network_module = import_module(args.network_config1, name="_network_module")
    network_options = {k: ast.literal_eval(v) for k, v in args.network_option}
    network_options.update(data_config.custom_model_kwargs)
    if args.gpus:
        gpus = [int(i) for i in args.gpus.split(",")]  # ?
        dev = torch.device(gpus[0])
    else:
        gpus = None
        local_rank = 0
        dev = torch.device("cpu")
    model, model_info = network_module.get_model(
        data_config, args=args, dev=dev, **network_options
    )

    if args.load_model_weights_1:
        print("Loading model state from %s" % args.load_model_weights_1)
        model_state = torch.load(args.load_model_weights_1, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        _logger.info(
            "Model initialized with weights from %s\n ... Missing: %s\n ... Unexpected: %s"
            % (args.load_model_weights, missing_keys, unexpected_keys)
        )

    loss_func = torch.nn.CrossEntropyLoss()

    return model, model_info, loss_func
