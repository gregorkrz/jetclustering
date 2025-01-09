import os
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import pickle
import torch
import time
from src.utils.utils import CPU_Unpickler
from src.dataset.get_dataset import get_iter
from src.utils.paths import get_path
from pathlib import Path


# This script attempts to open dataset files and prints the number of events in each one.
R = 0.8

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--dataset-cap", type=int, default=-1)
parser.add_argument("--output", type=str, default="")
parser.add_argument("--plot-only", action="store_true")
parser.add_argument("--jets-object", type=str, default="fatjets")
parser.add_argument("--eval-dir", type=str, default="")

args = parser.parse_args()
path = get_path(args.input, "preprocessed_data")

if args.eval_dir:
    eval_dir = get_path(args.eval_dir, "results")
    dataset_path_to_eval_file = {}
    for file in os.listdir(eval_dir):
        if file.startswith("eval_") and file.endswith(".pkl"):
            file_number = file.split("_")[1].split(".")[0]
            dataset_path_to_eval_file[CPU_Unpickler(open(os.path.join(eval_dir, file), "rb")).load()["filename"]] = [os.path.join(eval_dir, file), os.path.join(eval_dir, "clustering_{}.pkl".format(file_number))]
    print(dataset_path_to_eval_file)

if args.output == "":
    args.output = args.input

output_path = os.path.join(get_path(args.output, "results"), "count_matched_quarks")
Path(output_path).mkdir(parents=True, exist_ok=True)

def get_bc_scores_for_jets(event):
    scores = event.pfcands.bc_scores_pfcands
    clusters = event.pfcands.bc_labels_pfcands
    selected_clusters_idx = torch.where(event.model_jets.pt > 100)[0]
    result = []
    for c in selected_clusters_idx:
        result.append(scores[clusters == c.item()])
    return result

if not args.plot_only:
    n_matched_quarks = {}
    unmatched_quarks = {}
    n_fake_jets = {} # Number of jets that have not been matched to a quark
    bc_scores_matched = {}
    bc_scores_unmatched = {}
    for subdataset in os.listdir(path):
        print("-----", subdataset, "-----")
        current_path = os.path.join(path, subdataset)
        model_clusters_file = None
        model_output_file = None
        if args.eval_dir:
            model_clusters_file = dataset_path_to_eval_file[current_path][1]
            model_output_file = dataset_path_to_eval_file[current_path][0]
        dataset = get_iter(current_path, model_clusters_file=model_clusters_file, model_output_file=model_output_file,
                           include_model_jets_unfiltered=True)
        n = 0
        for data in tqdm(dataset):
            jets_object = data.__dict__[args.jets_object]
            n += 1
            if args.dataset_cap != -1 and n > args.dataset_cap:
                break
            jets = [jets_object.eta, jets_object.phi]
            dq = [data.matrix_element_gen_particles.eta, data.matrix_element_gen_particles.phi]
            # calculate deltaR between each jet and each quark
            distance_matrix = np.zeros((len(jets_object), len(data.matrix_element_gen_particles)))
            for i in range(len(jets_object)):
                for j in range(len(data.matrix_element_gen_particles)):
                    deta = jets[0][i] - dq[0][j]
                    dphi = jets[1][i] - dq[1][j]
                    distance_matrix[i, j] = np.sqrt(deta**2 + dphi**2)
            # row-wise argmin
            distance_matrix = distance_matrix.T
            #min_distance = np.min(distance_matrix, axis=1)
            n_jets = len(jets_object)
            if len(jets_object):
                quark_to_jet = np.min(distance_matrix, axis=1)
                quark_to_jet[quark_to_jet > R] = -1
                n_matched_quarks[subdataset] = n_matched_quarks.get(subdataset, []) + [np.sum(quark_to_jet != -1)]
                n_fake_jets[subdataset] = n_fake_jets.get(subdataset, []) + [n_jets - np.sum(quark_to_jet != -1)]
                filt = quark_to_jet == -1
                if args.jets_object == "model_jets":
                    matched_jet_idx = sorted(np.argmin(distance_matrix, axis=1)[quark_to_jet != -1])
                    unmatched_jet_idx = sorted(list(set(list(range(n_jets))) - set(matched_jet_idx)))
                    scores = get_bc_scores_for_jets(data)
                    for i in matched_jet_idx:
                        bc_scores_matched[subdataset] = bc_scores_matched.get(subdataset, []) + [torch.mean(scores[i]).item()]
                    for i in unmatched_jet_idx:
                        bc_scores_unmatched[subdataset] = bc_scores_unmatched.get(subdataset, []) + [torch.mean(scores[i]).item()]
            else:
                n_matched_quarks[subdataset] = n_matched_quarks.get(subdataset, []) + [0]
                n_fake_jets[subdataset] = n_fake_jets.get(subdataset, []) + [n_jets]
                filt = torch.ones(len(data.matrix_element_gen_particles)).bool()
                quark_to_jet = torch.ones(len(data.matrix_element_gen_particles)).long() * -1
            if subdataset not in unmatched_quarks:
                unmatched_quarks[subdataset] = {"pt": [], "eta": [], "phi": [], "pt_all": [], "frac_evt_E_matched": [], "frac_evt_E_unmatched": []}
            unmatched_quarks[subdataset]["pt"] += data.matrix_element_gen_particles.pt[filt].tolist()
            unmatched_quarks[subdataset]["pt_all"] += data.matrix_element_gen_particles.pt.tolist()
            unmatched_quarks[subdataset]["eta"] += data.matrix_element_gen_particles.eta[filt].tolist()
            unmatched_quarks[subdataset]["phi"] += data.matrix_element_gen_particles.phi[filt].tolist()
            visible_E_event = torch.sum(data.pfcands.E) #+ torch.sum(data.special_pfcands.E)
            matched_quarks = np.where(quark_to_jet != -1)[0]
            for i in range(len(data.matrix_element_gen_particles)):
                dq_coords = [dq[0][i], dq[1][i]]
                cone_filter = torch.sqrt((data.pfcands.eta - dq_coords[0])**2 + (data.pfcands.phi - dq_coords[1])**2) < R
                #cone_filter_special = torch.sqrt(
                #    (data.special_pfcands.eta - dq_coords[0]) ** 2 + (data.special_pfcands.phi - dq_coords[1]) ** 2) < R
                E_in_cone = data.pfcands.E[cone_filter].sum()# + data.special_pfcands.E[cone_filter_special].sum()
                if i in matched_quarks:
                    unmatched_quarks[subdataset]["frac_evt_E_matched"].append(E_in_cone / visible_E_event)
                else:
                    unmatched_quarks[subdataset]["frac_evt_E_unmatched"].append(E_in_cone / visible_E_event)
            #print("Number of matched quarks:", np.sum(quark_to_jet != -1))
    avg_n_matched_quarks = {}
    avg_n_fake_jets = {}
    for key in n_matched_quarks:
        avg_n_matched_quarks[key] = np.mean(n_matched_quarks[key])
        avg_n_fake_jets[key] = np.mean(n_fake_jets[key])
    def get_properties(name):
        # get mediator mass, dark quark mass, r_inv from the filename
        parts = name.split("_")
        mMed = int(parts[1].split("-")[1])
        mDark = int(parts[2].split("-")[1])
        rinv = float(parts[3].split("-")[1])
        return mMed, mDark, rinv
    result = {}
    result_unmatched = {}
    result_fakes = {}
    result_bc = {}
    for key in avg_n_matched_quarks:
        mMed, mDark, rinv = get_properties(key)
        if mMed not in result:
            result[mMed] = {}
            result_unmatched[mMed] = {}
            result_fakes[mMed] = {}
            result_bc[mMed] = {}
        if mDark not in result[mMed]:
            result[mMed][mDark] = {}
            result_unmatched[mMed][mDark] = {}
            result_fakes[mMed][mDark] = {}
            result_bc[mMed][mDark] = {}
        result[mMed][mDark][rinv] = avg_n_matched_quarks[key]
        result_unmatched[mMed][mDark][rinv] = unmatched_quarks[key]
        result_fakes[mMed][mDark][rinv] = avg_n_fake_jets[key]
        result_bc[mMed][mDark][rinv] = {
            "matched": bc_scores_matched[key],
            "unmatched": bc_scores_unmatched[key]
        }
    pickle.dump(result, open(os.path.join(output_path, "result.pkl"), "wb"))
    pickle.dump(result_unmatched, open(os.path.join(output_path, "result_unmatched.pkl"), "wb"))
    pickle.dump(result_fakes, open(os.path.join(output_path, "result_fakes.pkl"), "wb"))
    pickle.dump(result_bc, open(os.path.join(output_path, "result_bc.pkl"), "wb"))
if args.plot_only:
    result = pickle.load(open(os.path.join(output_path, "result.pkl"), "rb"))
    result_unmatched = pickle.load(open(os.path.join(output_path, "result_unmatched.pkl"), "rb"))
    result_fakes = pickle.load(open(os.path.join(output_path, "result_fakes.pkl"), "rb"))
    result_bc = pickle.load(open(os.path.join(output_path, "result_bc.pkl"), "rb"))
import matplotlib.pyplot as plt
# heatmap plots
mediator_masses = sorted(list(result.keys()))
dark_masses = [20]
r_invs = sorted(list(set([rinv for mMed in result for mDark in result[mMed] for rinv in result[mMed][mDark]])))
fig, ax = plt.subplots(len(dark_masses), 1, figsize=(5, 5))
if len(dark_masses) == 1:
    ax = [ax]
for i, mDark in enumerate(dark_masses):
    data = np.zeros((len(mediator_masses), len(r_invs)))
    for j, mMed in enumerate(mediator_masses):
        for k, rinv in enumerate(r_invs):
            data[j, k] = result[mMed][mDark][rinv]
    ax[i].imshow(data, cmap="Blues")
    for (j, k), val in np.ndenumerate(data):
        ax[i].text(k, j, f'{val:.2f}', ha='center', va='center', color='black')
    ax[i].set_xticks(range(len(r_invs)))
    ax[i].set_xticklabels(r_invs)
    ax[i].set_yticks(range(len(mediator_masses)))
    ax[i].set_yticklabels(mediator_masses)
    ax[i].set_xlabel("$r_{inv}$")
    ax[i].set_ylabel("$m_{Z'}$ [GeV]")
    ax[i].set_title(f"mDark = {mDark} GeV")
    cbar = fig.colorbar(ax[i].imshow(data, cmap="Blues"), ax=ax[i])
    cbar.set_label("Avg. matched dark quarks / event")
fig.tight_layout()
fig.savefig(os.path.join(output_path, "avg_matched_dark_quarks.pdf"))
print("Done")

fig, ax = plt.subplots(len(dark_masses), 1, figsize=(5, 5))
if len(dark_masses) == 1:
    ax = [ax]
for i, mDark in enumerate(dark_masses):
    data = np.zeros((len(mediator_masses), len(r_invs)))
    for j, mMed in enumerate(mediator_masses):
        for k, rinv in enumerate(r_invs):
            data[j, k] = result_fakes[mMed][mDark][rinv]
    ax[i].imshow(data, cmap="Blues")
    for (j, k), val in np.ndenumerate(data):
        ax[i].text(k, j, f'{val:.2f}', ha='center', va='center', color='black')
    ax[i].set_xticks(range(len(r_invs)))
    ax[i].set_xticklabels(r_invs)
    ax[i].set_yticks(range(len(mediator_masses)))
    ax[i].set_yticklabels(mediator_masses)
    ax[i].set_xlabel("$r_{inv}$")
    ax[i].set_ylabel("$m_{Z'}$ [GeV]")
    ax[i].set_title(f"mDark = {mDark} GeV")
    cbar = fig.colorbar(ax[i].imshow(data, cmap="Greens"), ax=ax[i])
    cbar.set_label("Avg. unmatched jets / event")
fig.tight_layout()
fig.savefig(os.path.join(output_path, "avg_unmatched_jets.pdf"))
print("Done")


fig, ax = plt.subplots(len(r_invs), len(mediator_masses), figsize=(3*len(mediator_masses), 3 * len(r_invs)))
for i in range(len(r_invs)):
    for j in range(len(mediator_masses)):
        data = result_unmatched[mediator_masses[j]][dark_masses[0]][r_invs[i]]["pt"]
        ax[i, j].hist(data, bins=50, histtype="step", label="Unmatched")
        ax[i, j].hist(result_unmatched[mediator_masses[j]][dark_masses[0]][r_invs[i]]["pt_all"], bins=50, histtype="step", label="All")
        ax[i, j].set_title(f"mMed = {mediator_masses[j]}, rinv = {r_invs[i]}")
        ax[i, j].set_xlabel("pt")
        ax[i, j].legend()

fig.tight_layout()
fig.savefig(os.path.join(output_path, "unmatched_dark_quarks_pt.pdf"))

fig, ax = plt.subplots(len(r_invs), len(mediator_masses), figsize=(3*len(mediator_masses), 3 * len(r_invs)))
for i in range(len(r_invs)):
    for j in range(len(mediator_masses)):
        data_x = result_unmatched[mediator_masses[j]][dark_masses[0]][r_invs[i]]["eta"]
        data_y = result_unmatched[mediator_masses[j]][dark_masses[0]][r_invs[i]]["phi"]
        # 2d histogram
        ax[i, j].hist2d(data_x, data_y, bins=10, cmap="Blues")
        ax[i, j].set_title(f"mMed = {mediator_masses[j]}, rinv = {r_invs[i]}")
        ax[i, j].set_xlabel("unmatched dark quark eta")
        ax[i, j].set_ylabel("unmatched dark quark phi")

fig.tight_layout()
fig.savefig(os.path.join(output_path, "unmatched_dark_quarks_eta_phi.pdf"))


fig, ax = plt.subplots(len(r_invs), len(mediator_masses), figsize=(3*len(mediator_masses), 3 * len(r_invs)))
for i in range(len(r_invs)):
    for j in range(len(mediator_masses)):
        data = result_unmatched[mediator_masses[j]][dark_masses[0]][r_invs[i]]["frac_evt_E_matched"]
        data_unmatched = result_unmatched[mediator_masses[j]][dark_masses[0]][r_invs[i]]["frac_evt_E_unmatched"]
        bins = np.linspace(0, 1, 100)
        ax[i, j].hist(data_unmatched, bins=bins, histtype="step", label="Unmatched")
        ax[i, j].hist(data, bins=bins, histtype="step", label="Matched")
        ax[i, j].set_title(f"mMed = {mediator_masses[j]}, rinv = {r_invs[i]}")
        ax[i, j].set_xlabel("E (R<0.8) / event E")
        ax[i, j].legend()
fig.tight_layout()
fig.savefig(os.path.join(output_path, "frac_E_in_cone.pdf"))

fig, ax = plt.subplots(len(r_invs), len(mediator_masses), figsize=(3*len(mediator_masses), 3 * len(r_invs)))
for i in range(len(r_invs)):
    for j in range(len(mediator_masses)):
        data = result_unmatched[mediator_masses[j]][dark_masses[0]][r_invs[i]]["frac_evt_E_matched"]
        data_unmatched = result_unmatched[mediator_masses[j]][dark_masses[0]][r_invs[i]]["frac_evt_E_unmatched"]
        bins = np.linspace(0, 1, 100)
        ax[i, j].hist(data_unmatched, bins=bins, histtype="step", label="Unmatched dark quark", density=True)
        ax[i, j].hist(data, bins=bins, histtype="step", label="Matched dark quark", density=True)
        ax[i, j].set_title(f"mMed = {mediator_masses[j]}, rinv = {r_invs[i]}")
        ax[i, j].set_xlabel("E (R<0.8) / event E")
        ax[i, j].legend()
fig.tight_layout()
fig.savefig(os.path.join(output_path, "frac_E_in_cone_density.pdf"))


fig, ax = plt.subplots(figsize=(5, 5))
unmatched = result_bc[900][20][0.3]["unmatched"]
matched = result_bc[900][20][0.3]["matched"]
bins = np.linspace(0, 1, 100)
ax.hist(unmatched, bins=bins, histtype="step", label="Unmatched jet")
ax.hist(matched, bins=bins, histtype="step", label="Matched jet")
ax.set_title("mMed = 900, mDark = 20, rinv = 0.3")
ax.set_xlabel("BC score")
ax.set_ylabel("count")
ax.set_yscale("log")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(output_path, "scores.pdf"))
