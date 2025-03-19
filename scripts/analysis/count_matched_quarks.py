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
from src.plotting.eval_matrix import matrix_plot
from src.utils.paths import get_path
from pathlib import Path
import matplotlib.pyplot as plt
from src.dataset.dataset import EventDataset

# This script attempts to open dataset files and prints the number of events in each one.
R = 0.8

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--dataset-cap", type=int, default=-1)
parser.add_argument("--output", type=str, default="")
parser.add_argument("--plot-only", action="store_true")
parser.add_argument("--jets-object", type=str, default="fatjets")
parser.add_argument("--eval-dir", type=str, default="")
parser.add_argument("--clustering-suffix", type=str, default="") # default: 1020, also want to try 1010 or others...?


parser.add_argument("--parton-level", "-pl", action="store_true") # To be used together with 'fastjet_jets'
parser.add_argument("--gen-level", "-gl", action="store_true")


args = parser.parse_args()
path = get_path(args.input, "preprocessed_data")

if args.eval_dir:
    eval_dir = get_path(args.eval_dir, "results", fallback=True)
    dataset_path_to_eval_file = {}
    for file in os.listdir(eval_dir):
        if file.startswith("eval_") and file.endswith(".pkl"):
            file_number = file.split("_")[1].split(".")[0]
            clustering_file = "clustering_{}.pkl".format(file_number)
            if args.clustering_suffix:
                clustering_file = "clustering_{}_{}.pkl".format(args.clustering_suffix, file_number)
            f = CPU_Unpickler(open(os.path.join(eval_dir, file), "rb")).load()
            clustering_file = os.path.join(eval_dir, clustering_file)
            if "model_cluster" in f and not args.clustering_suffix:
                clustering_file = None
            dataset_path_to_eval_file[f["filename"]] = [os.path.join(eval_dir, file), clustering_file]
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

def calculate_m(objects, mt=False):
    # set a mask returning only the two highest pt jets
    mask = objects.pt.argsort(descending=True)[:2]
    total_E = objects.E[mask].sum()
    total_pxyz = objects.pxyz[mask].sum(dim=0)
    if mt:
        return np.sqrt(total_E**2 - total_pxyz[0]**2 - total_pxyz[1]**2).item()
    return np.sqrt(total_E**2 - total_pxyz[2]**2 - total_pxyz[1]**2 - total_pxyz[0]**2).item()

thresholds = np.linspace(0.1, 1, 20)
# also add 100 points between 0 and 0.1 at the beginning
thresholds = np.concatenate([np.linspace(0, 0.1, 100), thresholds])

if not args.plot_only:
    n_matched_quarks = {}
    unmatched_quarks = {}
    n_fake_jets = {} # Number of jets that have not been matched to a quark
    bc_scores_matched = {}
    bc_scores_unmatched = {}
    precision_and_recall = {} # Array of [n_relevant_retrieved, all_retrieved, all_relevant], or in our language, [n_matched_dark_quarks, n_jets, n_dark_quarks]
    precision_and_recall_fastjets = {}
    pr_obj_score_thresholds = {} # same as precision_and_recall, except it gives a dictionary instead of the array, and the keys are the thresholds for objectness score
    mass_resolution = {} # Contains {'m_true': [], 'm_pred': [], 'mt_true': [], 'mt_pred': []} # mt = transverse mass, m = invariant mass
    for subdataset in os.listdir(path):
        print("-----", subdataset, "-----")
        current_path = os.path.join(path, subdataset)
        model_clusters_file = None
        model_output_file = None
        if subdataset not in precision_and_recall:
            precision_and_recall[subdataset] = [0, 0, 0]
            precision_and_recall_fastjets[subdataset] = {}
            pr_obj_score_thresholds[subdataset] = {}
            for i in range(len(thresholds)):
                pr_obj_score_thresholds[subdataset][i] = [0, 0, 0]
        if subdataset not in mass_resolution:
            mass_resolution[subdataset] = {'m_true': [], 'm_pred': [], 'mt_true': [], 'mt_pred': [], 'n_jets': []}
        if args.eval_dir:
            if current_path not in dataset_path_to_eval_file:
                print("Skipping", current_path)
                continue
            model_clusters_file = dataset_path_to_eval_file[current_path][1]
            model_output_file = dataset_path_to_eval_file[current_path][0]
        #dataset = get_iter(current_path, model_clusters_file=model_clusters_file, model_output_file=model_output_file,
        #                   include_model_jets_unfiltered=True)
        fastjet_R = None
        if args.jets_object == "fastjet_jets":
            fastjet_R = np.array([0.8, 2.0])
        dataset = EventDataset.from_directory(current_path, model_clusters_file=model_clusters_file,
                                    model_output_file=model_output_file,
                                    include_model_jets_unfiltered=True, fastjet_R=fastjet_R,
                                    parton_level=args.parton_level, gen_level=args.gen_level)
        n = 0
        for x in tqdm(range(len(dataset))):
            data = dataset[x]
            if data is None:
                print("Skipping", x)
                continue
            #try:
            #    data = dataset[x]
            #except:
            #    print("Exception")
            #    break # skip this event
            jets_object = data.__dict__[args.jets_object]
            n += 1
            if args.dataset_cap != -1 and n > args.dataset_cap:
                break
            if not args.jets_object == "fastjet_jets":
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
                precision_and_recall[subdataset][1] += n_jets
                precision_and_recall[subdataset][2] += len(data.matrix_element_gen_particles)
                if "obj_score" in jets_object.__dict__:
                    print("Also evaluating using objectness score")
                    for i in range(len(thresholds)):
                        filt = torch.sigmoid(jets_object.obj_score) >= thresholds[i]
                        pr_obj_score_thresholds[subdataset][i][1] += torch.sum(filt).item()
                        pr_obj_score_thresholds[subdataset][i][2] += len(data.matrix_element_gen_particles)
                mass_resolution[subdataset]['m_true'].append(calculate_m(data.matrix_element_gen_particles))
                mass_resolution[subdataset]['m_pred'].append(calculate_m(jets_object))
                mass_resolution[subdataset]['mt_true'].append(calculate_m(data.matrix_element_gen_particles, mt=True))
                mass_resolution[subdataset]['mt_pred'].append(calculate_m(jets_object, mt=True))
                mass_resolution[subdataset]['n_jets'].append(n_jets)
                if len(jets_object):
                    quark_to_jet = np.min(distance_matrix, axis=1)
                    quark_to_jet[quark_to_jet > R] = -1
                    n_matched_quarks[subdataset] = n_matched_quarks.get(subdataset, []) + [np.sum(quark_to_jet != -1)]
                    n_fake_jets[subdataset] = n_fake_jets.get(subdataset, []) + [n_jets - np.sum(quark_to_jet != -1)]
                    precision_and_recall[subdataset][0] += np.sum(quark_to_jet != -1)
                    if "obj_score" in jets_object.__dict__:
                        for i in range(len(thresholds)):
                            filt = torch.sigmoid(jets_object.obj_score) >= thresholds[i]
                            dist_matrix_filt = distance_matrix[:, filt.numpy()]
                            if filt.sum() == 0:
                                continue
                            quark_to_jet_filt = np.min(dist_matrix_filt, axis=1)
                            quark_to_jet_filt[quark_to_jet_filt > R] = -1
                            pr_obj_score_thresholds[subdataset][i][0] += np.sum(quark_to_jet_filt != -1)
                    filt = quark_to_jet == -1
                    #if args.jets_object == "model_jets":
                        #matched_jet_idx = sorted(np.argmin(distance_matrix, axis=1)[quark_to_jet != -1])
                        #unmatched_jet_idx = sorted(list(set(list(range(n_jets))) - set(matched_jet_idx)))
                        #scores = get_bc_scores_for_jets(data)
                        #for i in matched_jet_idx:
                        #    bc_scores_matched[subdataset] = bc_scores_matched.get(subdataset, []) + [torch.mean(scores[i]).item()]
                        #for i in unmatched_jet_idx:
                        #    bc_scores_unmatched[subdataset] = bc_scores_unmatched.get(subdataset, []) + [torch.mean(scores[i]).item()]
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
            else:
                for key in jets_object:
                    jets = [jets_object[key].eta, jets_object[key].phi]
                    dq = [data.matrix_element_gen_particles.eta, data.matrix_element_gen_particles.phi]
                    # calculate deltaR between each jet and each quark
                    distance_matrix = np.zeros((len(jets_object[key]), len(data.matrix_element_gen_particles)))
                    for i in range(len(jets_object[key])):
                        for j in range(len(data.matrix_element_gen_particles)):
                            deta = jets[0][i] - dq[0][j]
                            dphi = jets[1][i] - dq[1][j]
                            distance_matrix[i, j] = np.sqrt(deta ** 2 + dphi ** 2)
                    # row-wise argmin
                    distance_matrix = distance_matrix.T
                    # min_distance = np.min(distance_matrix, axis=1)
                    n_jets = len(jets_object[key])
                    if key not in precision_and_recall_fastjets[subdataset]:
                        precision_and_recall_fastjets[subdataset][key] = [0, 0, 0]
                    precision_and_recall_fastjets[subdataset][key][1] += n_jets
                    precision_and_recall_fastjets[subdataset][key][2] += len(data.matrix_element_gen_particles)
                    if len(jets_object[key]):
                        quark_to_jet = np.min(distance_matrix, axis=1)
                        quark_to_jet[quark_to_jet > R] = -1
                        precision_and_recall_fastjets[subdataset][key][0] += np.sum(quark_to_jet != -1)
    avg_n_matched_quarks = {}
    avg_n_fake_jets = {}
    for key in n_matched_quarks:
        avg_n_matched_quarks[key] = np.mean(n_matched_quarks[key])
        avg_n_fake_jets[key] = np.mean(n_fake_jets[key])
    def get_properties(name):
        # get mediator mass, dark quark mass, r_inv from the filename
        parts = name.strip().strip("/").split("/")[-1].split("_")
        try:
            mMed = int(parts[1].split("-")[1])
            mDark = int(parts[2].split("-")[1])
            rinv = float(parts[3].split("-")[1])
        except:
            # another convention
            mMed = int(parts[2].split("-")[1])
            mDark = int(parts[3].split("-")[1])
            rinv = float(parts[4].split("-")[1])
        return mMed, mDark, rinv
    result = {}
    result_unmatched = {}
    result_fakes = {}
    result_bc = {}
    result_PR = {}
    result_PR_AKX = {}
    result_PR_thresholds = {}
    result_m = {}
    if args.jets_object != "fastjet_jets":
        for key in avg_n_matched_quarks:
            mMed, mDark, rinv = get_properties(key)
            if mMed not in result:
                result[mMed] = {}
                result_unmatched[mMed] = {}
                result_fakes[mMed] = {}
                result_bc[mMed] = {}
                result_PR[mMed] = {}
                result_PR_AKX[mMed] = {}
                result_PR_thresholds[mMed] = {}
                result_m[mMed] = {}
            if mDark not in result[mMed]:
                result[mMed][mDark] = {}
                result_unmatched[mMed][mDark] = {}
                result_fakes[mMed][mDark] = {}
                result_bc[mMed][mDark] = {}
                result_PR[mMed][mDark] = {}
                result_PR_thresholds[mMed][mDark] = {}
                result_PR_AKX[mMed][mDark] = {}
                result_m[mMed][mDark] = {}
            result[mMed][mDark][rinv] = avg_n_matched_quarks[key]
            result_unmatched[mMed][mDark][rinv] = unmatched_quarks[key]
            result_fakes[mMed][mDark][rinv] = avg_n_fake_jets[key]
            #result_bc[mMed][mDark][rinv] = {
            #    "matched": bc_scores_matched[key],
            #    "unmatched": bc_scores_unmatched[key]
            #}
            result_PR_thresholds[mMed][mDark][rinv] = pr_obj_score_thresholds[key]
            if  precision_and_recall[key][1] == 0 or precision_and_recall[key][2] == 0:
                result_PR[mMed][mDark][rinv] = [0, 0]
                print(mMed, mDark, rinv)
                print("PR zero", key, precision_and_recall[key])
            else:
                result_PR[mMed][mDark][rinv] = [precision_and_recall[key][0] / precision_and_recall[key][1], precision_and_recall[key][0] / precision_and_recall[key][2]]
            result_m[mMed][mDark][rinv] = {key: np.array(val) for key, val in mass_resolution[key].items()}
            if args.jets_object == "fastjet_jets":
                r = precision_and_recall_fastjets[key]
                if rinv not in result_PR_AKX[mMed][mDark]:
                    result_PR_AKX[mMed][mDark][rinv] = {}
                for k in r:
                    if r[k][1] == 0 or r[k][2] == 0:
                        result_PR_AKX[mMed][mDark][rinv][k] = [0, 0]
                    else:
                        result_PR_AKX[mMed][mDark][rinv][k] = [r[k][0] / r[k][1], r[k][0] / r[k][2]]
    else:
        for key in precision_and_recall_fastjets:
            mMed, mDark, rinv = get_properties(key)
            if mMed not in result_PR_AKX:
                result_PR_AKX[mMed] = {}
            if mDark not in result_PR_AKX[mMed]:
                result_PR_AKX[mMed][mDark] = {}
            r = precision_and_recall_fastjets[key]
            if rinv not in result_PR_AKX[mMed][mDark]:
                result_PR_AKX[mMed][mDark][rinv] = {}
            for k in r:
                if r[k][1] == 0 or r[k][2] == 0:
                    result_PR_AKX[mMed][mDark][rinv][k] = [0, 0]
                else:
                    result_PR_AKX[mMed][mDark][rinv][k] = [r[k][0] / r[k][1], r[k][0] / r[k][2]]
    pickle.dump(result, open(os.path.join(output_path, "result.pkl"), "wb"))
    pickle.dump(result_unmatched, open(os.path.join(output_path, "result_unmatched.pkl"), "wb"))
    pickle.dump(result_fakes, open(os.path.join(output_path, "result_fakes.pkl"), "wb"))
    pickle.dump(result_bc, open(os.path.join(output_path, "result_bc.pkl"), "wb"))
    if args.jets_object == "fastjet_jets":
        pickle.dump(result_PR_AKX, open(os.path.join(output_path, "result_PR_AKX.pkl"), "wb"))
    pickle.dump(result_PR, open(os.path.join(output_path, "result_PR.pkl"), "wb"))
    pickle.dump(result_PR_thresholds, open(os.path.join(output_path, "result_PR_thresholds.pkl"), "wb"))
    pickle.dump(result_m, open(os.path.join(output_path, "result_m.pkl"), "wb"))
    with open(os.path.join(output_path, "eval_done.txt"), "w") as f:
        f.write("True")
    # Write the number of events to n_events.txt
    with open(os.path.join(output_path, "n_events.txt"), "w") as f:
        f.write(str(n))
if args.plot_only:
    result = pickle.load(open(os.path.join(output_path, "result.pkl"), "rb"))
    result_unmatched = pickle.load(open(os.path.join(output_path, "result_unmatched.pkl"), "rb"))
    result_fakes = pickle.load(open(os.path.join(output_path, "result_fakes.pkl"), "rb"))
    result_bc = pickle.load(open(os.path.join(output_path, "result_bc.pkl"), "rb"))
    result_PR = pickle.load(open(os.path.join(output_path, "result_PR.pkl"), "rb"))
    result_PR_thresholds = pickle.load(open(os.path.join(output_path, "result_PR_thresholds.pkl"), "rb"))

if args.jets_object == "fastjet_jets":
    print("Only computing fastjet jets - exiting now, the metrics have been saved to disk")
    import sys
    sys.exit(0)

fig, ax = plt.subplots(3, 1, figsize=(4, 12))

def get_plots_for_params(mMed, mDark, rInv):
    precisions = []
    recalls = []
    f1_scores = []
    for i in range(len(thresholds)):
        if result_PR_thresholds[mMed][mDark][rInv][i][1] == 0:
            precisions.append(0)
        else:
            precisions.append(result_PR_thresholds[mMed][mDark][rInv][i][0] / result_PR_thresholds[mMed][mDark][rInv][i][1])
        if result_PR_thresholds[mMed][mDark][rInv][i][2] == 0:
            recalls.append(0)
        else:
            recalls.append(result_PR_thresholds[mMed][mDark][rInv][i][0] / result_PR_thresholds[mMed][mDark][rInv][i][2])
    for i in range(len(thresholds)):
        if precisions[i] + recalls[i] == 0:
            f1_scores.append(0)
        else:
            f1_scores.append(2*precisions[i]*recalls[i] / (precisions[i] + recalls[i]))
    return precisions, recalls, f1_scores


def plot_for_params(a, b, c):
    precisions, recalls, f1_scores = get_plots_for_params(a, b, c)
    ax[0].plot(thresholds, precisions, ".--", label=f"mMed={a},rInv={c}")
    ax[1].plot(thresholds, recalls, ".--", label=f"mMed={a},rInv={c}")
    ax[2].plot(thresholds, f1_scores, ".--", label=f"mMed={a},rInv={c}")

plot_for_params(900, 20, 0.3)
plot_for_params(700, 20, 0.7)
plot_for_params(3000, 20, 0.3)
plot_for_params(900, 20, 0.7)
plot_for_params(1000, 20, 0.3)
ax[0].grid()
ax[1].grid()
ax[2].grid()
ax[0].set_ylabel("Precision")
ax[1].set_ylabel("Recall")
ax[2].set_ylabel("F1 score")
ax[0].legend()
ax[1].legend()
ax[2].legend()
ax[0].set_xscale("log")
ax[1].set_xscale("log")
ax[2].set_xscale("log")
fig.tight_layout()
fig.savefig(os.path.join(output_path, "pr_thresholds.pdf"))

matrix_plot(result, "Blues", "Avg. matched dark quarks / event").savefig(os.path.join(output_path, "avg_matched_dark_quarks.pdf"))
matrix_plot(result_fakes, "Greens", "Avg. unmatched jets / event").savefig(os.path.join(output_path, "avg_unmatched_jets.pdf"))
matrix_plot(result_PR, "Reds", "Precision (N matched dark quarks / N predicted jets)", metric_comp_func = lambda r: r[0]).savefig(os.path.join(output_path, "precision.pdf"))
matrix_plot(result_PR, "Reds", "Recall (N matched dark quarks / N dark quarks)", metric_comp_func = lambda r: r[1]).savefig(os.path.join(output_path, "recall.pdf"))
matrix_plot(result_PR, "Purples", "F_1 score", metric_comp_func = lambda r: 2 * r[0] * r[1] / (r[0] + r[1])).savefig(os.path.join(output_path, "f1_score.pdf"))

dark_masses = [20]
mediator_masses = sorted(list(result.keys()))
r_invs = sorted(list(set([rinv for mMed in result for mDark in result[mMed] for rinv in result[mMed][mDark]])))

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

'''
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
fig.savefig(os.path.join(output_path, "avg_scores_matched_vs_unmatched_jet.pdf"))
'''