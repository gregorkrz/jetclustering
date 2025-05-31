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
from src.layers.object_cond import calc_eta_phi
torch.autograd.set_detect_anomaly(True)
from src.dataset.functions_data import get_batch
from src.dataset.functions_data import concat_events, Event, EventPFCands
from src.plotting.plot_event import plot_event
from src.dataset.dataset import EventDataset
from src.jetfinder.clustering import get_clustering_labels
from torch_scatter import scatter_sum

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

import fastjet

def inference(loss_str, train_dataset_str, input_text, input_text_quarks):
    args = argparse.ArgumentParser()
    model_path = f"models/{loss_str}/{train_dataset_str}.ckpt"
    args.spatial_part_only = True # LGATr
    args.load_model_weights =  model_path
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

    # input text in format pt,eta,phi,mass,charge
    pt, eta, phi, mass, charge = [], [], [], [], []
    # now parse the input text
    for line in input_text.strip().split('\n'):
        values = list(map(float, line.split()))
        pt.append(values[0])
        eta.append(values[1])
        phi.append(values[2])
        mass.append(values[3])
        charge.append(int(values[4]))
    pt_quarks, eta_quarks, phi_quarks = [], [], []
    for line in input_text_quarks.strip().split("\n"):
        values = list(map(float, line.split()))
        pt_quarks.append(values[0])
        eta_quarks.append(values[1])
        phi_quarks.append(values[2])
    pid = torch.zeros(len(pt))
    pf_cand_jet_idx = [-1] * len(pt)

    pfcands = EventPFCands(pt, eta, phi, mass, charge, pid, pf_cand_jet_idx=pf_cand_jet_idx)
    n_soft = 0
    if "GP" in loss_str:
        n_soft = 500
    if n_soft > 0:
        pfcands = EventDataset.pfcands_add_soft_particles(pfcands, n_soft, random_generator=np.random.RandomState(seed=0))
    event = Event(pfcands=pfcands)
    event_batch = concat_events([event])
    batch, _ = get_batch(event_batch, batch_config, torch.zeros(len(pfcands)), test=True)

    with torch.no_grad():
        coords = model(batch, cpu_demo=True)[:, 1:4] # !!! Only use cpu_demo with batch size of 1 (quick fix for unavailability of xformers attention on CPU)
    clust_labels = get_clustering_labels(coords.detach().cpu().numpy(), batch.batch_idx, min_cluster_size=args.min_cluster_size, min_samples=args.min_samples, epsilon=args.epsilon)
    jets_pxyz = scatter_sum(torch.tensor(pfcands.pxyz), torch.tensor(clust_labels+1), dim=0)[1:]
    jets_pt = torch.norm(jets_pxyz[:, :2], p=2, dim=-1)
    filt = torch.where(jets_pt > 30)[0].tolist()
    jets_eta, jets_phi = calc_eta_phi(jets_pxyz, False)
    clust_assignment = {}
    for i in range(len(clust_labels)):
        if clust_labels[i] in filt and clust_labels[i] != -1:
            clust_assignment[i] = filt.index(clust_labels[i])
    jets_pt = jets_pt[filt]
    jets_eta = jets_eta[filt]
    jets_phi = jets_phi[filt]
    ak_pt, ak_eta, ak_phi, _, ak_assignment = EventDataset.get_jets_fastjets_raw_with_assignment(pfcands, fastjet.JetDefinition(fastjet.antikt_algorithm, 0.8), pt_cutoff=30)
    model_coords = calc_eta_phi(coords, return_stacked=0)
    clist = ['#1f78b4', '#b3df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbe6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99',
             '#b15928']
    colors = {
        -1: "gray",
        0: clist[0],
        1: clist[1],
        2: clist[2],
        3: clist[3],
        4: clist[4],
        5: clist[5],
        6: clist[6],
        7: clist[7],
    }
    c = []
    c_ak = []
    for i in range(len(pfcands)):
        if i in ak_assignment:
            c_ak.append(colors.get(ak_assignment[i], "purple"))
        else:
            c_ak.append("gray")
        if i in clust_assignment:
            c.append(colors.get(clust_assignment[i], "gray"))
        else:
            c.append("gray")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(10, 3.33)) # with AK colors, with model colors, with model colors in clustering space
    ax[0].set_title("Colors: AK clusters")
    ax[1].set_title("Colors: Model clusters")
    ax[2].set_title("Colors: Model clusters in cl. space")
    plot_event(event, colors=c_ak, ax=ax[0], jets=0)
    plot_event(event, colors=c, ax=ax[1], jets=0)
    plot_event(event, colors=c, ax=ax[2], custom_coords=model_coords, jets=0)
    model_jets, ak_jets = [], []

    for j in range(len(ak_pt)):
        if ak_pt[j] >= 30:
            ax[0].text(ak_eta[j] + 0.1, ak_phi[j] + 0.1,
                                   "pt=" + str(round(ak_pt[j], 1)), color="blue", fontsize=6, alpha=0.5)
            ak_jets.append({"pt": ak_pt[j], "eta": ak_eta[j], "phi": ak_phi[j]})
        if ak_pt[j] >= 100:
            for k in range(3):
                circle = plt.Circle((ak_eta[j], ak_phi[j]), 0.8, color="green", fill=False, alpha=.7)
                ax[k].add_artist(circle)


    for j in range(len(jets_pt)):
        if jets_pt[j] >= 30:
            ax[1].text(jets_eta[j] + 0.1, jets_phi[j] + 0.1,
                                   "pt=" + str(round(jets_pt[j].item(), 1)), color="gray", fontsize=6, alpha=0.5)
            model_jets.append({"pt": jets_pt[j].item(), "eta": jets_eta[j].item(), "phi": jets_phi[j].item()})

        if jets_pt[j] >= 100:
            for k in range(3):
                circle = plt.Circle((jets_eta[j], jets_phi[j]), 0.7, color="blue", fill=False, alpha=.7)
                ax[k].add_artist(circle)
    for k in range(3):
        #for n in range(len(phi_quarks)):
        #   # add triangle symb
        ax[k].scatter(eta_quarks, phi_quarks, s=pt_quarks,  c="red", marker="^", alpha=0.3)
        ax[k].set_xlabel("$\eta$")
        ax[k].set_ylabel("$\phi$")
    fig.tight_layout()
    return model_jets, ak_jets, fig

