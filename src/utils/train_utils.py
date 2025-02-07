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
from src.layers.object_cond import calc_eta_phi
from src.layers.object_cond import object_condensation_loss


def to_filelist(args, mode="train"):
    if mode == "train":
        flist = args.data_train
    elif mode == "val":
        flist = args.data_val
    elif mode == "test":
        flist = args.data_test
    else:
        raise NotImplementedError("Invalid mode %s" % mode)
    print(mode, "filelist:", flist)
    flist = [get_path(p, "preprocessed_data") for p in flist]
    return flist

class TensorCollection:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    def to(self, device):
        # Move all tensors to device
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                setattr(self, k, v.to(device))
        return self
    def dict_rep(self):
        d = {}
        for k, v in self.__dict__.items():
            if torch.is_tensor(v):
                d[k] = v
        return d
    #def __getitem__(self, i):
    #    return TensorCollection(**{k: v[i] for k, v in self.__dict__.items()})

def train_load(args):
    train_files = to_filelist(args, "train")
    val_files = to_filelist(args, "val")
    train_data = EventDatasetCollection(train_files)
    if args.train_dataset_size is not None:
        train_data = torch.utils.data.Subset(train_data, list(range(args.train_dataset_size)))
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=concat_events,
        persistent_workers=args.num_workers > 0,
        shuffle=False
    )
    '''val_loaders = {}
    for filename in val_files:
        val_data = EventDataset.from_directory(filename, mmap=True)
        val_loaders[filename] = DataLoader(
            val_data,
            batch_size=args.batch_size,
            drop_last=True,
            pin_memory=True,
            collate_fn=concat_events,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )'''
    val_data = EventDatasetCollection(val_files)
    if args.val_dataset_size is not None:
        val_data = torch.utils.data.Subset(val_data, list(range(args.val_dataset_size)))
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=concat_events,
        persistent_workers=args.num_workers > 0,
        shuffle=False
    )
    return train_loader, val_loader, val_data

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
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )
    return test_loaders

def get_optimizer_and_scheduler(args, model, device, load_model_weights="load_model_weights"):
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

    if args.__dict__[load_model_weights] is not None:
        _logger.info("Resume training from file %s" % args.__dict__[load_model_weights])
        model_state = torch.load(
            args.__dict__[load_model_weights],
            map_location=device,
        )
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model.module.load_state_dict(model_state["model"])
        else:
            model.load_state_dict(model_state["model"])
        opt_state = model_state["optimizer"]
        opt.load_state_dict(opt_state)
    scheduler = None
    if args.lr_scheduler == "steps":
        lr_step = round(args.num_epochs / 3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt,
            milestones=[10],
            gamma=0.20,
            last_epoch=-1
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
                last_epoch=-1,
                verbose=True,
            )
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                opt,
                milestones=milestones,
                gamma=gamma,
                last_epoch=-1
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
    if args.__dict__[load_model_weights] is not None:
       scheduler.load_state_dict(model_state["scheduler"])
    return opt, scheduler


def get_target_obj_score(clusters_eta, clusters_phi, clusters_pt, event_idx_clusters, dq_eta, dq_phi, dq_event_idx):
    # return the target scores for each cluster (reteurns list of 1's and 0's)
    # dq_coords: list of [eta, phi] for each dark quark
    # dq_event_idx: list of event_idx for each dark quarks
    target = []
    for event in event_idx_clusters.unique():
        filt = event_idx_clusters == event
        clusters = torch.stack([clusters_eta[filt], clusters_phi[filt], clusters_pt[filt]], dim=1)
        dq_coords_event = torch.stack([dq_eta[dq_event_idx == event], dq_phi[dq_event_idx == event]], dim=1)
        dist_matrix = torch.cdist(
            dq_coords_event,
            clusters[:, :2].to(dq_coords_event.device),
            p=2
        ).T
        if len(dist_matrix) == 0:
            target.append(torch.zeros(len(clusters)).int().to(dist_matrix.device))
            continue
        closest_dq_dist, closest_dq_idx = dist_matrix.min(dim=0)
        closest_cluster_dist, closest_cluster_idx = dist_matrix.min(dim=1)
        #print("Clusters", clusters[:, :2])
        #print("dist_matrix", dist_matrix)
        #print("Closest DQ dist", closest_dq_dist)
        #print("Dq coords event")
        #print(dq_coords_event)
        #print("Clusters pt", clusters_pt[filt])
        # for each dark quark, the closest cluster that is within 0.8 distance gets target set to 1 or otherwise to 0. If there are two clsuters in radius 0.8 around the dark quark, we only select one!!
        #target.append((closest_dq_dist < 0.8).float())
        #target.append((closest_dq_dist < 0.8).float())
        closest_quark_dist, closest_quark_idx = dist_matrix.min(dim=1)
        closest_quark_idx[closest_quark_dist > 0.8] = -1
        target.append((closest_quark_idx != -1).float())
    return torch.cat(target).flatten()

def plot_obj_score_debug(dq_eta, dq_phi, dq_batch_idx, clusters_eta, clusters_phi, clusters_pt, clusters_batch_idx, clusters_labels, input_pxyz, input_event_idx, input_clusters, pred_obj_score_clusters):
    # For debugging the Objectness Score head.
    import matplotlib.pyplot as plt
    n_events = dq_batch_idx.max().int().item() + 1
    pfcands_pt = torch.sqrt(input_pxyz[:, 0] ** 2 + input_pxyz[:, 1] ** 2)
    pfcands_eta, pfcands_phi = calc_eta_phi(input_pxyz, return_stacked=0)
    fig, ax = plt.subplots(1, n_events, figsize=(n_events * 3, 3))
    colors = {0: "grey", 1: "green"}
    for i in range(n_events):
        # Plot the clusters as dots that are green for label 1 and gray for label 0
        filt = clusters_batch_idx == i
        ax[i].scatter(clusters_eta[filt].cpu(), clusters_phi[filt].cpu(), c=[colors[x] for x in clusters_labels[filt].tolist()], cmap="coolwarm", s=clusters_pt[filt].cpu(), alpha=0.5)
        # with a light gray text, also plot the target objectness score for each cluster
        for j in range(len(clusters_eta[filt])):
            ax[i].text(clusters_eta[filt][j].cpu()-0.5, clusters_phi[filt][j].cpu()-0.5, str(round(pred_obj_score_clusters[filt][j].item(), 2)), fontsize=6, color="gray", alpha=0.7)
        # Plot the dark quarks as red dots
        filt = dq_batch_idx == i
        ax[i].scatter(dq_eta[filt].cpu(), dq_phi[filt].cpu(), c="red", alpha=0.5)
        ax[i].scatter(pfcands_eta[input_event_idx == i].cpu(), pfcands_phi[input_event_idx == i].cpu(), c=input_clusters[input_event_idx == i].cpu(), cmap="coolwarm", s=pfcands_pt[input_event_idx == i].cpu(), alpha=0.5)
        # put pt of the clusters in gray text on top of them
        filt = clusters_batch_idx == i
        for j in range(len(clusters_eta[filt])):
            ax[i].text(clusters_eta[filt][j].cpu(), clusters_phi[filt][j].cpu(), str(round(clusters_pt[filt][j].item(), 2)), fontsize=8, color="black")

    fig.tight_layout()
    return fig


def get_loss_func(args):
    # Loss function  takes in the output of a model and the output of GT (the GT labels) and returns the loss.
    def loss(model_input, model_output, gt_labels):
            batch_numbers = model_input.batch_idx
            if not (args.loss == "quark_distance" or args.train_objectness_score):
                labels = gt_labels+1
            else:
                labels = gt_labels
            return object_condensation_loss(model_input, model_output, labels, batch_numbers,
                                            attr_weight=args.attr_loss_weight,
                                            repul_weight=args.repul_loss_weight,
                                            coord_weight=args.coord_loss_weight,
                                            beta_type=args.beta_type,
                                            lorentz_norm=args.lorentz_norm,
                                            spatial_part_only=args.spatial_part_only,
                                            loss_quark_distance=args.loss=="quark_distance",
                                            oc_scalars=args.scalars_oc,
                                            loss_obj_score=args.train_objectness_score)
    return loss


def renumber_clusters(tensor):
    unique = tensor.unique()
    mapping = torch.zeros(unique.max() + 1)
    for i, u in enumerate(unique):
        mapping[u] = i
    return mapping[tensor]

def get_gt_func(args):
    # Gets the GT function: the function accepts an Event batch
    # and returns the ground truth labels (GT idx of a dark quark it belongs to, or -1 for noise)
    # By default, it returns the dark quark that is closest to the event, IF it's closer than R.
    R = args.gt_radius
    def get_idx_for_event(obj, i):
        return obj.batch_number[i], obj.batch_number[i + 1]
    def get_labels(b, pfcands, special=False, get_coordinates=False, get_dq_coords=False):
        # b: Batch of events
        # if get_coordinates is true, it returns the coordinates of the labels rather than the clustering labels themselves.
        labels = torch.zeros(len(pfcands)).long()
        if get_coordinates:
            labels_coordinates = torch.zeros(len(b.matrix_element_gen_particles.pt), 4).float()
            labels_no_renumber = torch.ones_like(labels)*-1
            offset = 0
        if get_dq_coords:
            dq_coords = [b.matrix_element_gen_particles.eta, b.matrix_element_gen_particles.phi]
            #dq_coords_batch_idx = b.matrix_element_gen_particles.batch_number
            dq_coords_batch_idx = torch.zeros(b.matrix_element_gen_particles.pt.shape)
            for i in range(len(b.matrix_element_gen_particles.batch_number) - 1):
                dq_coords_batch_idx[b.matrix_element_gen_particles.batch_number[i]:b.matrix_element_gen_particles.batch_number[i + 1]] = i
        for i in range(len(b)):
            s_dq, e_dq = get_idx_for_event(b.matrix_element_gen_particles, i)
            dq_eta = b.matrix_element_gen_particles.eta[s_dq:e_dq]
            dq_phi = b.matrix_element_gen_particles.phi[s_dq:e_dq]
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
            if len(closest_quark_idx):
                if special: print("Closest quark idx", closest_quark_idx, "; renumbered ",
                                  renumber_clusters(closest_quark_idx + 1) - 1)
                if not get_coordinates:
                    closest_quark_idx = renumber_clusters(closest_quark_idx + 1) - 1
                else:
                    labels_no_renumber[s:e] = closest_quark_idx
                    closest_quark_idx[closest_quark_idx != -1] += offset
            labels[s:e] = closest_quark_idx
            if get_coordinates:
                E_dq = b.matrix_element_gen_particles.E[s_dq:e_dq]
                pxyz_dq = b.matrix_element_gen_particles.pxyz[s_dq:e_dq] # the -1 doesn't matter as it will be ignored anyway
                labels_coordinates[s_dq:e_dq] = torch.cat([E_dq.unsqueeze(-1), pxyz_dq], dim=1)
                offset += len(E_dq)
        if get_coordinates:
            return TensorCollection(labels=labels, labels_coordinates=labels_coordinates, labels_no_renumber=labels_no_renumber)
        if get_dq_coords:
            return TensorCollection(labels=labels, dq_coords=dq_coords, dq_coords_batch_idx=dq_coords_batch_idx)
        return labels
    def gt(events):
        #special_labels = get_labels(events, events.special_pfcands, special=True)
        #print("Special pfcands labels", special_labels)
        #return torch.cat([get_labels(events, events.pfcands), special_labels])
        return get_labels(events, events.pfcands, get_coordinates=args.loss=="quark_distance",
                          get_dq_coords=args.train_objectness_score)
    return gt


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model(args, dev):
    network_options = {}  # TODO: implement network options
    network_module = import_module(args.network_config, name="_network_module")
    model = network_module.get_model(obj_score=False, args=args, **network_options)
    if args.load_model_weights:
        print("Loading model state dict from %s" % args.load_model_weights)
        model_state = torch.load(args.load_model_weights, map_location=dev)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        _logger.info(
            "Model initialized with weights from %s\n ... Missing: %s\n ... Unexpected: %s"
            % (args.load_model_weights, missing_keys, unexpected_keys)
        )
        assert len(missing_keys) == 0
        assert len(unexpected_keys) == 0
    return model

def get_model_obj_score(args, dev):
    network_options = {}  # TODO: implement network options
    network_module = import_module("src/models/transformer/transformer.py", name="_network_module")
    model = network_module.get_model(obj_score=True, args=args, **network_options)
    if args.load_objectness_score_weights:
        assert args.train_objectness_score
        print("Loading objectness score model state dict from %s" % args.load_objectness_score_weights)
        model_state = torch.load(args.load_objectness_score_weights, map_location=dev)["model"]
        missing_keys, unexpected_keys = model.load_state_dict(model_state, strict=False)
        _logger.info(
            "Objectness score model initialized with weights from %s\n ... Missing: %s\n ... Unexpected: %s"
            % (args.load_objectness_score_weights, missing_keys, unexpected_keys)
        )
        assert len(missing_keys) == 0
        assert len(unexpected_keys) == 0
    return model
