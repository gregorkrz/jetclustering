import os
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import pickle
from src.dataset.get_dataset import get_iter
from src.utils.paths import get_path
from pathlib import Path
import torch
# This script attempts to open dataset files and prints the number of events in each one.

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--dataset-cap", type=int, default=-1)
parser.add_argument("--output", type=str, default="")
parser.add_argument("--plot-only", action="store_true")

# Plots of stats: total visible energy, visible mass, number of AK8 jets, number of pfcands + special_pfcands

args = parser.parse_args()
path = get_path(args.input, "preprocessed_data")
if args.output == "":
    args.output = args.input
output_path = os.path.join(get_path(args.output, "results"), "dataset_stats")
Path(output_path).mkdir(parents=True, exist_ok=True)

if not args.plot_only:
    stats = {}
    for subdataset in os.listdir(path):
        print("-----", subdataset, "-----")
        current_path = os.path.join(path, subdataset)
        dataset = get_iter(current_path)
        n = 0
        stats[subdataset] = {"total_visible_E": [], "visible_mass": [], "n_fatjets": [], "n_pfcands": []}
        for data in tqdm(dataset):
            n += 1
            if args.dataset_cap != -1 and n > args.dataset_cap:
                break
            n_fatjets = len(data.fatjets)
            n_pfcands = len(data.pfcands) + len(data.special_pfcands)
            total_visible_E = torch.sum(data.pfcands.E) + torch.sum(data.special_pfcands.E)
            visible_mass = torch.sqrt(torch.sum(data.pfcands.E)**2 - torch.sum(data.pfcands.p)**2)
            stats[subdataset]["total_visible_E"].append(total_visible_E)
            stats[subdataset]["visible_mass"].append(visible_mass)
            stats[subdataset]["n_fatjets"].append(n_fatjets)
            stats[subdataset]["n_pfcands"].append(n_pfcands)
        #stats[subdataset]["n_events"] = dataset.n_events
    def get_properties(name):
        # get mediator mass, dark quark mass, r_inv from the filename
        parts = name.split("_")
        mMed = int(parts[1].split("-")[1])
        mDark = int(parts[2].split("-")[1])
        rinv = float(parts[3].split("-")[1])
        return mMed, mDark, rinv
    result = {}
    for key in stats:
        mMed, mDark, rinv = get_properties(key)
        if mMed not in result:
            result[mMed] = {}
        if mDark not in result[mMed]:
            result[mMed][mDark] = {}
        result[mMed][mDark][rinv] = stats[key]
    pickle.dump(result, open(os.path.join(output_path, "result.pkl"), "wb"))
if args.plot_only:
    result = pickle.load(open(os.path.join(output_path, "result.pkl"), "rb"))

import matplotlib.pyplot as plt
# heatmap plots
mediator_masses = sorted(list(result.keys()))
dark_masses = [20]
r_invs = sorted(list(set([rinv for mMed in result for mDark in result[mMed] for rinv in result[mMed][mDark]])))

def plot_distribution(result, key_name):
    fig, ax = plt.subplots(len(mediator_masses), len(r_invs), figsize=(3*len(r_invs), 3*len(mediator_masses)))
    for i, mMed in enumerate(mediator_masses):
        for j, rinv in enumerate(r_invs):
            mDark = dark_masses[0]
            data = result[mMed][mDark][rinv][key_name]
            ax[i, j].hist(data, bins=50)
            ax[i, j].set_title(f"$m_{{Z'}}$={mMed},$r_{{inv}}$={rinv} ($\Sigma$={int(sum(data))})")
    # big title
    fig.suptitle(key_name)
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, f"{key_name}.pdf"))
    #fig.show()

plot_distribution(result, "total_visible_E")
plot_distribution(result, "visible_mass")
plot_distribution(result, "n_fatjets")
plot_distribution(result, "n_pfcands")



