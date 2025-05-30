# A simple wrapper to run the L-GATr model on HuggingFace spaces
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
from src.dataset.functions_data import get_batch
from src.dataset.functions_data import concat_events, Event, EventPFCands
from src.jetfinder.clustering import get_clustering_labels

from src.utils.train_utils import (
    to_filelist,
    train_load,
    test_load,
    get_model,
    get_optimizer_and_scheduler,
    get_model_obj_score
)
from src.utils.paths import get_path
import warnings
import pickle
import os


args = argparse.ArgumentParser()
args.spatial_part_only = True # LGATr
args.load_model_weights =  get_path("train/Delphes_Aug_IRCSplit_50k_SN_from3kFT_2025_05_16_14_07_29_474/step_21060_epoch_2.ckpt", "results", fallback=True) # for debugging
args.aug_soft = True # LGATr_GP etc.
args.network_config = "src/models/LGATr/lgatr.py"
args.beta_type = "pt+bc"
args.embed_as_vectors = False
args.debug = False
args.epsilon = 0.3
args.gen_level = False
args.parton_level = False
args.global_features_obj_score = False
args.gt_radius = 0.8
args.no_pid = True
args.hidden_mv_channels = 16
args.hidden_s_channels = 64
args.internal_dim = 128
args.lorentz_norm = False
args.min_cluster_size = 2
args.min_samples = 1
args.n_heads = 4
args.num_blocks = 10
args.scalars_oc=False

dev = torch.device("cpu")
model = get_model(args, dev)
orig_model = model
batch_config = {"use_p_xyz": True, "use_four_momenta": False}

if "lgatr" in args.network_config.lower():
    batch_config = {"use_four_momenta": True}
batch_config["no_pid"] = True

print("batch_config:", batch_config)
model.eval()


# INPUTS
pt = [1, 2, 3]
eta = [0.5, 0.6, 0.7]
phi = [0, 2, -1]
charge = [0, -1, 1]
mass = [0.511, 0, 0]

pid = torch.zeros(len(pt))
pf_cand_jet_idx = [-1] * len(pt)

pfcands = EventPFCands(pt, eta, phi, mass, charge, pid, pf_cand_jet_idx=pf_cand_jet_idx)
event = Event(pfcands=pfcands)
event_batch = concat_events([event])
batch, _ = get_batch(event_batch, batch_config, torch.zeros(len(pt)), test=True)

coords = model(batch, cpu_demo=True)[:, 1:4] # !!! Only use cpu_demo with batch size of 1 (quick fix for unavailability of xformers attention on CPU)
clust_labels = get_clustering_labels(coords, batch.batch_idx, min_cluster_size=args.min_cluster_size, min_samples=args.min_samples, epsilon=args.epsilon, )

