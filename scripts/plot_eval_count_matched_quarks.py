import torch
from itertools import chain, combinations

import os
from tqdm import tqdm
import argparse
import pickle
from src.plotting.eval_matrix import matrix_plot, scatter_plot, multiple_matrix_plot, ax_tiny_histogram
from src.utils.paths import get_path
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

#### Plotting functions

from matplotlib_venn import venn3
import matplotlib.pyplot as plt
from copy import copy


def plot_venn3_from_index_dict(ax, data_dict, set_labels=('Set 0', 'Set 1', 'Set 2'), set_colors=("orange", "purple", "gray"), remove_max=True):
    """
    Generate a 3-set Venn diagram from a dictionary where keys are strings of '0', '1', and '2'
    indicating set membership, and values are counts.

    Parameters:
    - data_dict (dict): Dictionary with keys like '0', '01', '012', etc.
    - set_labels (tuple): Labels for the three sets.
    - remove_max: if true, it will remove
    """
    # Mapping of set index combinations to venn3 region codes
    index_to_region = {
        '100': '100',  # Only in Set 0
        '010': '010',  # Only in Set 1
        '001': '001',  # Only in Set 2
        '110': '110',  # In Set 0 and Set 1
        '101': '101',  # In Set 0 and Set 2
        '011': '011',  # In Set 1 and Set 2
        '111': '111',  # In all three
    }

    # Initialize region counts
    venn_counts = {region: 0 for region in index_to_region.values()}
    max_value = 0
    for key in data_dict:
        if data_dict[key] > max_value and key != "":
            max_value = data_dict[key]
    print("Max val", max_value)
    data_dict = copy(data_dict)
    new_data_dict = {}
    for key in data_dict:
        if remove_max and data_dict[key] == max_value:
        #    #data_dict[key] = 0
        #    del data_dict[key]
            continue
        else:
            new_data_dict[key] = data_dict[key]
    data_dict = new_data_dict
    print("data dict", data_dict)
    # Convert data_dict keys to binary keys for region mapping
    for k, v in data_dict.items():
        binary_key = ''.join(['1' if str(i) in k else '0' for i in range(3)])
        if binary_key in index_to_region:
            venn_counts[index_to_region[binary_key]] += v
    # Plotting
    #plt.figure(figsize=(8, 8))
    del venn_counts['111']
    venn = venn3(subsets=venn_counts, set_labels=set_labels, set_colors=set_colors, alpha=0.5, ax=ax)
    venn.get_label_by_id("111").set_text(max_value)
    #plt.title("3-Set Venn Diagram from Index Dictionary")
    #plt.show()


### Change this to make custom plots highlighting differences between different models (the histograms of pt_pred/pt_true, eta_pred-eta_true, and phi_pred-phi_true)
histograms_dict = {
        "": [{"base_LGATr": 50000, "base_Tr": 50000 , "base_GATr": 50000, "AK8": 50000}, {"base_LGATr": "orange", "base_Tr": "blue", "base_GATr": "green", "AK8": "gray"}],
        "LGATr_comparison": [{"base_LGATr": 50000, "LGATr_GP_IRC_S_50k": 9960, "LGATr_GP_50k": 9960, "AK8": 50000, "LGATr_GP_IRC_SN_50k": 24000}, {"base_LGATr": "orange", "LGATr_GP_IRC_S_50k": "red", "LGATr_GP_50k": "purple", "LGATr_GP_IRC_SN_50k": "pink", "AK8": "gray"}],
        "LGATr_comparison_QCDtrain": [{"LGATr_QCD": 50000, "LGATr_GP_IRC_S_QCD": 9960, "LGATr_GP_QCD": 24000, "AK8": 50000,
                              "LGATr_GP_IRC_SN_QCD": 24000},
                             {"LGATr_QCD": "orange", "LGATr_GP_IRC_S_QCD": "red", "LGATr_GP_QCD": "purple",
                              "LGATr_GP_IRC_SN_QCD": "pink", "AK8": "gray"}],
    "LGATr_comparison_DifferentTrainingDS": [{"base_LGATr": 50000, "LGATr_700_07": 50000, "LGATr_QCD": 50000, "LGATr_700_07+900_03": 50000, "LGATr_700_07+900_03+QCD": 50000, "AK8": 50000}, {"base_LGATr": "orange", "LGATr_700_07": "red", "LGATr_QCD": "purple", "LGATr_700_07+900_03": "blue", "LGATr_700_07+900_03+QCD": "green", "AK8": "gray"}]
}

# This is a dictionary that contains the models and their colors for plotting - to plot the F1 scores etc. of the models
results_dict = {
    "LGATr_comparison_DifferentTrainingDS":
        [{"base_LGATr": "orange", "LGATr_700_07": "red", "LGATr_QCD": "purple", "LGATr_700_07+900_03": "blue",
         "LGATr_700_07+900_03+QCD": "green", "AK8": "gray"}, {"base_LGATr": "LGATr_900_03"}], # 2nd dict in list is rename dict
    "LGATr_comparison": [{"base_LGATr": "orange", "LGATr_GP_IRC_S_50k": "red", "LGATr_GP_50k": "purple", "LGATr_GP_IRC_SN_50k": "pink", "AK8": "gray"},
                         {"base_LGATr": "LGATr", "LGATr_GP_IRC_S_50k": "LGATr_GP_IRC_S", "LGATr_GP_50k": "LGATr_GP", "LGATr_GP_IRC_SN_50k": "LGATr_GP_IRC_SN"}], # 2nd dict in list is rename dict
    "LGATr_comparison_QCDtrain": [{"LGATr_QCD": "orange", "LGATr_GP_IRC_S_QCD": "red", "LGATr_GP_QCD": "purple", "LGATr_GP_IRC_SN_QCD": "pink", "AK8": "gray"},
                         {"LGATr_QCD": "LGATr", "LGATr_GP_IRC_S_QCD": "LGATr_GP_IRC_S", "LGATr_GP_QCD": "LGATr_GP", "LGATr_GP_IRC_SN_QCD": "LGATr_GP_IRC_SN"}], # 2nd dict in list is rename dict
    "LGATr_comparison_GP_training": [
        {"LGATr_GP_QCD": "purple", "LGATr_GP_700_07": "red", "LGATr_GP_700_07+900_03": "blue", "LGATr_GP_700_07+900_03+QCD": "green",  "LGATr_GP_50k": "orange", "AK8": "gray"},
        {"LGATr_GP_QCD": "QCD", "LGATr_GP_700_07": "700_07", "LGATr_GP_700_07+900_03": "700_07+900_03" ,  "LGATr_GP_50k": "900_03", "LGATr_GP_700_07+900_03+QCD": "700_07+900_03+QCD"} # 2nd dict in list is rename dict
    ],
    "LGATr_comparison_GP_IRC_S_training": [
        {"LGATr_GP_IRC_S_QCD": "purple", "LGATr_GP_IRC_S_700_07": "red", "LGATr_GP_IRC_S_700_07+900_03": "blue", "LGATr_GP_IRC_S_700_07+900_03+QCD": "green", "LGATr_GP_IRC_S_50k": "orange", "AK8": "gray"},
        {"LGATr_GP_IRC_S_QCD": "QCD", "LGATr_GP_IRC_S_700_07": "700_07", "LGATr_GP_IRC_S_700_07+900_03": "700_07+900_03",  "LGATr_GP_IRC_S_50k": "900_03",  "LGATr_GP_IRC_S_700_07+900_03+QCD": "700_07+900_03+QCD"} # 2nd dict in list is rename dict
    ],
    "LGATr_comparison_GP_IRC_SN_training": [
        {"LGATr_GP_IRC_SN_QCD": "purple", "LGATr_GP_IRC_SN_700_07": "red", "LGATr_GP_IRC_SN_700_07+900_03": "blue", "LGATr_GP_IRC_SN_700_07+900_03+QCD": "green", "LGATr_GP_IRC_SN_50k": "orange", "AK8": "gray"},
        {"LGATr_GP_IRC_SN_QCD": "QCD", "LGATr_GP_IRC_SN_700_07": "700_07", "LGATr_GP_IRC_SN_700_07+900_03": "700_07+900_03",  "LGATr_GP_IRC_SN_50k": "900_03",  "LGATr_GP_IRC_SN_700_07+900_03+QCD": "700_07+900_03+QCD"} # 2nd dict in list is rename dict
    ]
}


'''
        "GP_LGATr_training_NoPID_Delphes_PU_PFfix_QCD_events_10_16_64_0.8_2025_05_19_21_29_06_946": "LGATr_GP_QCD",
        "GP_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_10_16_64_0.8_2025_05_19_21_38_20_376": "LGATr_GP_700_07",
        "GP_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_AND_900_03_AND_QCD_10_16_64_0.8_2025_05_20_13_12_54_359": "LGATr_GP_700_07+900_03+QCD",
        "GP_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_AND_900_03_10_16_64_0.8_2025_05_20_13_13_00_503": "LGATr_GP_700_07+900_03",
        "GP_IRC_S_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_AND_900_03_10_16_64_0.8_2025_05_20_15_29_30_29": "LGATr_GP_IRC_S_700_07+900_03",
        "GP_IRC_S_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_AND_900_03_AND_QCD_10_16_64_0.8_2025_05_20_15_29_28_959": "LGATr_GP_IRC_S_700_07+900_03+QCD",
        "GP_IRC_S_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_10_16_64_0.8_2025_05_20_15_11_35_476": "LGATr_GP_IRC_S_700_07",
        "GP_IRC_S_LGATr_training_NoPID_Delphes_PU_PFfix_QCD_events_10_16_64_0.8_2025_05_20_15_11_20_735": "LGATr_GP_IRC_S_QCD",

'''

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=False, default="scouting_PFNano_signals2/SVJ_hadronic_std/batch_eval/objectness_score")
parser.add_argument("--threshold-obj-score", "-os-threshold", type=float, default=-1)

thresholds = np.linspace(0.1, 1, 20)
# also add 100 points between 0 and 0.1 at the beginning
thresholds = np.concatenate([np.linspace(0, 0.1, 100), thresholds])

args = parser.parse_args()
path = get_path(args.input, "results")

models = sorted([x for x in os.listdir(path) if not os.path.isfile(os.path.join(path, x))])
models = [x for x in models if "AKX" not in x]

figures_all = {} # title to the f1 score figure to plot
figures_all_sorted = {} # model used -> step -> level -> f1 figure
print("Models:", models)

radius = {
    "LGATr_R10": 1.0,
    "LGATr_R09": 0.9,
    "LGATr_rinv_03_m_900": 0.8,
    "LGATr_R06": 0.6,
    "LGATr_R07": 0.7,
    "LGATr_R11": 1.1,
    "LGATr_R12": 1.2,
    "LGATr_R13": 1.3,
    "LGATr_R14": 1.4,
    "LGATr_R20": 2.0,
    "LGATr_R25": 2.5
}

comments = {
    "Eval_params_study_2025_02_17_13_30_50": ", tr. on 07_700",
    "Eval_objectness_score_2025_02_12_15_34_33": ", tr. on 03_900, GT=all",
    "Eval_objectness_score_2025_02_18_08_48_13": ", tr. on 03_900, GT=closest",
    "Eval_objectness_score_2025_02_14_11_10_14": ", tr. on 03_900, GT=closest",
    "Eval_objectness_score_2025_02_21_14_51_07": ", tr. on 07_700",
    "Eval_objectness_score_2025_02_10_14_59_49": ", tr. on 03_900, GT=all",
    "Eval_objectness_score_2025_02_23_19_26_25": ", tr. on all, GT=closest",
    "Eval_objectness_score_2025_02_23_21_04_33": ", tr. on 03_900, GT=closest"
}

out_file_PR = os.path.join(get_path(args.input, "results"), "precision_recall.pdf")
out_file_PRf1 = os.path.join(get_path(args.input, "results"), "f1_score_sorted.pdf")

out_file_PG = os.path.join(get_path(args.input, "results"), "PLoverGL.pdf")

if args.threshold_obj_score != -1:
    out_file_PR_OS = os.path.join(get_path(args.input, "results"), f"precision_recall_with_obj_score.pdf")

out_file_avg_number_matched_quarks = os.path.join(get_path(args.input, "results"), "avg_number_matched_quarks.pdf")

def get_plots_for_params(mMed, mDark, rInv, result_PR_thresholds):
    precisions = []
    recalls = []
    f1_scores = []
    for i in range(len(thresholds)):
        if result_PR_thresholds[mMed][mDark][rInv][i][1] == 0:
            precisions.append(0)
        else:
            precisions.append(
                result_PR_thresholds[mMed][mDark][rInv][i][0] / result_PR_thresholds[mMed][mDark][rInv][i][1])
        if result_PR_thresholds[mMed][mDark][rInv][i][2] == 0:
            recalls.append(0)
        else:
            recalls.append(
                result_PR_thresholds[mMed][mDark][rInv][i][0] / result_PR_thresholds[mMed][mDark][rInv][i][2])
    for i in range(len(thresholds)):
        if precisions[i] + recalls[i] == 0:
            f1_scores.append(0)
        else:
            f1_scores.append(2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i]))
    return precisions, recalls, f1_scores

sz = 5
nplots = 9

# Now make 3 plots, one for mMed=700,r_inv=0.7; one for mMed=700,r_inv=0.5; one for mMed=700,r_inv=0.3
###fig, ax = plt.subplots(3, 3, figsize=(3 * sz, 3 * sz))

'''fig, ax = plt.subplots(3, nplots, figsize=(nplots*sz, 3*sz))
for mi, mass in enumerate([700, 900, 1500]):
    start_idx = mi*3
    for i0, rinv in enumerate([0.3, 0.5, 0.7]):
        i = start_idx + i0
        # 0 is precision, 1 is recall, 2 is f1 score
        ax[0, i].set_title(f"r_inv={rinv}, m_med={mass} GeV")
        ax[1, i].set_title(f"r_inv={rinv}, m_med={mass} GeV")
        ax[2, i].set_title(f"r_inv={rinv}, m_med={mass} GeV")
        ax[0, i].set_ylabel("Precision")
        ax[1, i].set_ylabel("Recall")
        ax[2, i].set_ylabel("F1 score")
        ax[0, i].grid()
        ax[1, i].grid()
        ax[2, i].grid()
        ylims = {} # key: j and i
        default_ylims = [1, 0]
        for j, model in enumerate(models):
            result_PR_thresholds = os.path.join(path, model, "count_matched_quarks", "result_PR_thresholds.pkl")
            if not os.path.exists(result_PR_thresholds):
                continue
            run_config = pickle.load(open(os.path.join(path, model, "run_config.pkl"), "rb"))
            result_PR_thresholds = pickle.load(open(result_PR_thresholds, "rb"))
            if mass not in result_PR_thresholds:
                continue
            if rinv not in result_PR_thresholds[mass]:
                continue6
            precisions, recalls, f1_scores = get_plots_for_params(mass, 20, rinv, result_PR_thresholds)
            if not run_config["gt_radius"] == 0.8:
                continue
            label = "R={} gl.f.={} {}".format(run_config["gt_radius"], run_config.get("global_features_obj_score", False), comments.get(run_config["run_name"], run_config["run_name"]))
            scatter_plot(ax[0, i], thresholds, precisions, label=label)
            scatter_plot(ax[1, i], thresholds, recalls, label=label)
            scatter_plot(ax[2, i], thresholds, f1_scores, label=label)
            #ylims[0] = [min(ylims[0][0], min(precisions)), max(ylims[0][1], max(precisions))]
            #ylims[1] = [min(ylims[1][0], min(recalls)), max(ylims[1][1], max(recalls))]
            #ylims[2] = [min(ylims[2][0], min(f1_scores)), max(ylims[2][1], max(f1_scores))]
            filt = thresholds < 0.2
            precisions = np.array(precisions)[filt]
            recalls = np.array(recalls)[filt]
            f1_scores = np.array(f1_scores)[filt]
            if (i, 0) not in ylims:
                ylims[(i, 0)] = default_ylims
            upper_factor = 1.01
            lower_factor = 0.99
            ylims[(i, 0)] = [min(ylims[(i, 0)][0], min(precisions)*lower_factor), max(ylims[(i, 0)][1], max(precisions)*upper_factor)]
            if (i, 1) not in ylims:
                ylims[(i, 1)] = default_ylims
            ylims[(i, 1)] = [min(ylims[(i, 1)][0], min(recalls)*lower_factor), max(ylims[(i, 1)][1], max(recalls)*upper_factor)]
            if (i, 2) not in ylims:
                ylims[(i, 2)] = default_ylims
            ylims[(i, 2)] = [min(ylims[(i, 2)][0], min(f1_scores)*lower_factor), max(ylims[(i, 2)][1], max(f1_scores)*upper_factor)]
        for j in range(3):
            ax[j, i].set_ylim(ylims[(i, j)])
            ax[j, i].legend()
            ax[j, i].set_xlim([0, 0.2])
            ax[j, i].set_xlim([0, 0.2])
            ax[j, i].set_xlim([0, 0.2])
        # now adjust the ylim so that the plots are more readable

fig.tight_layout()
fig.savefig(os.path.join(get_path(args.input, "results"), "precision_recall_thresholds.pdf"))

print("Saved to", os.path.join(get_path(args.input, "results"), "precision_recall_thresholds.pdf"))'''


import wandb
api = wandb.Api()

def get_run_by_name(name):
    clust_suffix = ""
    if name.endswith("FT"):
        #remove FT from the end
        name = name[:-2]
        clust_suffix = "FT"
    if name.endswith("FT1"):
        #remove FT from the end # min-samples 1 min-cluster-size 2 epsilon 0.3
        name = name[:-3]
        clust_suffix = "FT1"
    if name.endswith("10_5"):
        name = name[:-4]
        clust_suffix = "10_5"
    runs = api.runs(
        path="fcc_ml/svj_clustering",
        filters={"display_name": {"$eq": name.strip()}}
    )
    runs = api.runs(
        path="fcc_ml/svj_clustering",
        filters={"display_name": {"$eq": name.strip()}}
    )
    if runs.length != 1:
        return None
    return runs[0], clust_suffix


def get_run_config(run_name):
    r, clust_suffix = get_run_by_name(run_name)
    if r is None:
        print("Getting info from run", run_name, "failed")
        return None, None
    config = r.config
    result = {}
    if config["parton_level"]:
        prefix = "PL"
        result["level"] = "PL"
        result["level_idx"] = 0
    elif config["gen_level"]:
        prefix = "GL"
        result["level"] = "GL"
        result["level_idx"] = 2
    else:
        prefix = "sc."
        result["level"] = "scouting"
        result["level_idx"] = 1
    if config["augment_soft_particles"]:
        result["ghosts"] = True
        #result["level"] += "+ghosts"
    gt_r = config["gt_radius"]
    if config.get("augment_soft_particles", False):
        prefix += " (aug)" # ["LGATr_training_NoPID_10_16_64_0.8_Aug_Finetune_vanishing_momentum_QCap05_2025_03_28_17_12_25_820", "LGATr_training_NoPID_10_16_64_2.0_Aug_Finetune_vanishing_momentum_QCap05_2025_03_28_17_12_26_400"]

    training_datasets = {
        "LGATr_training_NoPID_10_16_64_0.8_AllData_2025_02_28_13_42_59": "all",
        "LGATr_training_NoPID_10_16_64_0.8_2025_02_28_12_42_59": "900_03",
        "LGATr_training_NoPID_10_16_64_2.0_2025_02_28_12_48_58": "900_03",
        "LGATr_training_NoPID_10_16_64_0.8_700_07_2025_02_28_13_01_59": "700_07",
        "LGATr_training_NoPIDGL_10_16_64_0.8_2025_03_17_20_05_04": "900_03_GenLevel",
        "LGATr_training_NoPIDGL_10_16_64_2.0_2025_03_17_20_05_04": "900_03_GenLevel",
        "Transformer_training_NoPID_10_16_64_2.0_2025_03_03_17_00_38": "900_03_T",
        "Transformer_training_NoPID_10_16_64_0.8_2025_03_03_15_55_50": "900_03_T",
        "LGATr_training_NoPID_10_16_64_0.8_Aug_Finetune_2025_03_27_12_46_12_740": "900_03+SoftAug",
        "LGATr_training_NoPID_10_16_64_2.0_Aug_Finetune_vanishing_momentum_2025_03_28_10_43_36_81": "900_03+SoftAugVM",
        "LGATr_training_NoPID_10_16_64_0.8_Aug_Finetune_vanishing_momentum_2025_03_28_10_43_37_44": "900_03+SoftAugVM",
        "LGATr_training_NoPID_10_16_64_0.8_Aug_Finetune_vanishing_momentum_QCap05_2025_03_28_17_12_25_820": "900_03+qcap05",
        "LGATr_training_NoPID_10_16_64_2.0_Aug_Finetune_vanishing_momentum_QCap05_2025_03_28_17_12_26_400": "900_03+qcap05",
        "LGATr_training_NoPID_10_16_64_2.0_Aug_Finetune_vanishing_momentum_QCap05_1e-2_2025_03_29_14_58_38_650": "pt 1e-2",
        "LGATr_training_NoPID_10_16_64_0.8_Aug_Finetune_vanishing_momentum_QCap05_1e-2_2025_03_29_14_58_36_446": "pt 1e-2",
        "LGATr_pt_1e-2_500part_2025_04_01_16_49_08_406": "500_pt_1e-2_PLFT",
        "LGATr_pt_1e-2_500part_2025_04_01_21_14_07_350": "500_pt_1e-2_PLFT",
        "LGATr_pt_1e-2_500part_NoQMin_2025_04_03_23_15_17_745": "500_1e-2_scFT",
        "LGATr_pt_1e-2_500part_NoQMin_2025_04_03_23_15_35_810": "500_1e-2_scFT",
        "LGATr_pt_1e-2_500part_NoQMin_10_to_1000p_2025_04_04_12_57_51_536": "10_1000_1e-2_scFT",
        "LGATr_pt_1e-2_500part_NoQMin_10_to_1000p_2025_04_04_12_57_47_788": "10_1000_1e-2_scFT",
        "LGATr_pt_1e-2_500part_NoQMin_10_to_1000p_CW0_2025_04_04_15_30_16_839": "10_1000_1e-2_CW0",
        "LGATr_pt_1e-2_500part_NoQMin_10_to_1000p_CW0_2025_04_04_15_30_20_113": "10_1000_1e-2_CW0",
        "debug_IRC_loss_weighted100_plus_ghosts_2025_04_08_22_40_33_972": "IRC_short_debug",
        "debug_IRC_loss_weighted100_plus_ghosts_2025_04_09_13_48_55_569": "IRC",
        "debug_IRC_loss_weighted100_plus_ghosts_Qmin05_2025_04_09_14_45_51_381": "IRC_qmin05",
        "LGATr_500part_NOQMin_2025_04_09_21_53_37_210": "500part_NOQMin_reprod",
        "IRC_loss_Split_and_Noise_alternate_Aug_2025_04_14_11_10_21_788": "IRC_Aug_S+N",
        "IRC_loss_Split_and_Noise_alternate_NoAug_2025_04_11_16_15_48_955": "IRC_S+N",
        "LGATr_training_NoPID_Delphes_10_16_64_0.8_2025_04_17_18_07_38_405": "DelphesTrain",
        "Delphes_IRC_aug_2025_04_19_11_16_17_130": "DelphesTrain+IRC",
        "LGATr_500part_NOQMin_Delphes_2025_04_19_11_15_24_417": "DelphesTrain+ghosts",
        "Delphes_IRC_aug_SplitOnly_2025_04_20_15_50_33_553": "DelphesTrain+IRC_SplitOnly",
        "Delphes_IRC_NOAug_SplitOnly_2025_04_21_12_58_36_99": "Delphes_IRC_NoAug_SplitOnly",
        "Delphes_IRC_NOAug_SplitAndNoise_2025_04_21_19_32_08_865": "Delphes_IRC_NoAug_S+N",
        "CONT_Delphes_IRC_aug_SplitOnly_2025_04_21_12_53_27_730": "IRC_aug_SplitOnly_ContFrom14k",
        "Transformer_training_NoPID_Delphes_PU_10_16_64_0.8_2025_05_03_18_37_01_188": "base_Tr_Old",
        "LGATr_training_NoPID_Delphes_PU_PFfix_10_16_64_0.8_2025_05_03_18_35_53_134": "base_LGATr",
        "GATr_training_NoPID_Delphes_PU_10_16_64_0.8_2025_05_03_18_35_48_163": "base_GATr_Old",
        "Transformer_training_NoPID_Delphes_PU_CoordFix_10_16_64_0.8_2025_05_05_13_05_20_755": "base_Tr",
        "GATr_training_NoPID_Delphes_PU_CoordFix_SmallDS_10_16_64_0.8_2025_05_05_16_24_13_579": "base_GATr_SD",
        "GATr_training_NoPID_Delphes_PU_CoordFix_10_16_64_0.8_2025_05_05_13_06_27_898": "base_GATr",
        "LGATr_Aug_2025_05_06_10_08_05_956": "LGATr_GP",
        "Delphes_Aug_IRCSplit_CONT_2025_05_07_11_00_18_422": "LGATr_GP_IRC_S",
        "Delphes_Aug_IRC_Split_and_Noise_2025_05_07_14_43_13_968": "LGATr_GP_IRC_SN",
        "Transformer_training_NoPID_Delphes_PU_CoordFix_SmallDS_10_16_64_0.8_2025_05_05_16_24_19_936": "base_Tr_SD",
        "LGATr_training_NoPID_Delphes_PU_PFfix_SmallDS_10_16_64_0.8_2025_05_05_16_24_16_127": "base_LGATr_SD",
        "Delphes_Aug_IRCSplit_2025_05_06_10_09_00_567": "LGATr_GP_IRC_S",
        "GATr_training_NoPID_Delphes_PU_CoordFix_SmallDS_10_16_64_0.8_2025_05_09_15_34_13_531": "base_GATr_SD",
        "Transformer_training_NoPID_Delphes_PU_CoordFix_SmallDS_10_16_64_0.8_2025_05_09_15_56_50_216": "base_Tr_SD",
        "LGATr_training_NoPID_Delphes_PU_PFfix_SmallDS_10_16_64_0.8_2025_05_09_15_56_50_875": "base_LGATr_SD",
        "Delphes_Aug_IRCSplit_50k_from10k_2025_05_11_14_08_49_675": "LGATr_GP_IRC_S_50k",
        "LGATr_Aug_50k_2025_05_09_15_25_32_34": "LGATr_GP_50k",
        "Delphes_Aug_IRCSplit_50k_2025_05_09_15_22_38_956": "LGATr_GP_IRC_S_50k",
        "LGATr_training_NoPID_Delphes_PU_PFfix_700_07_AND_900_03_AND_QCD_10_16_64_0.8_2025_05_16_21_04_26_937": "LGATr_700_07+900_03+QCD",
        "LGATr_training_NoPID_Delphes_PU_PFfix_700_07_AND_900_03_10_16_64_0.8_2025_05_16_21_04_26_991": "LGATr_700_07+900_03",
        "LGATr_training_NoPID_Delphes_PU_PFfix_QCD_events_10_16_64_0.8_2025_05_16_19_46_57_48": "LGATr_QCD",
        "LGATr_training_NoPID_Delphes_PU_PFfix_700_07_10_16_64_0.8_2025_05_16_19_44_46_795": "LGATr_700_07",
        "Delphes_Aug_IRCSplit_50k_SN_from3kFT_2025_05_16_14_07_29_474": "LGATr_GP_IRC_SN_50k",
        "GP_LGATr_training_NoPID_Delphes_PU_PFfix_QCD_events_10_16_64_0.8_2025_05_19_21_29_06_946": "LGATr_GP_QCD",
        "GP_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_10_16_64_0.8_2025_05_19_21_38_20_376": "LGATr_GP_700_07",
        "GP_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_AND_900_03_AND_QCD_10_16_64_0.8_2025_05_20_13_12_54_359": "LGATr_GP_700_07+900_03+QCD",
        "GP_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_AND_900_03_10_16_64_0.8_2025_05_20_13_13_00_503": "LGATr_GP_700_07+900_03",
        "GP_IRC_S_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_AND_900_03_10_16_64_0.8_2025_05_20_15_29_30_29": "LGATr_GP_IRC_S_700_07+900_03",
        "GP_IRC_S_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_AND_900_03_AND_QCD_10_16_64_0.8_2025_05_20_15_29_28_959": "LGATr_GP_IRC_S_700_07+900_03+QCD",
        "GP_IRC_S_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_10_16_64_0.8_2025_05_20_15_11_35_476": "LGATr_GP_IRC_S_700_07",
        "GP_IRC_S_LGATr_training_NoPID_Delphes_PU_PFfix_QCD_events_10_16_64_0.8_2025_05_20_15_11_20_735": "LGATr_GP_IRC_S_QCD",
        "GP_IRC_S_LGATr_training_NoPID_Delphes_PU_PFfix_QCD_events_10_16_64_0.8_2025_05_24_23_00_54_948": "LGATr_GP_IRC_SN_QCD",
        "GP_IRC_S_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_AND_900_03_AND_QCD_10_16_64_0.8_2025_05_24_23_00_56_910": "LGATr_GP_IRC_SN_700_07+900_03+QCD",
        "GP_IRC_S_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_AND_900_03_10_16_64_0.8_2025_05_24_23_01_01_212": "LGATr_GP_IRC_SN_700_07+900_03",
        "GP_IRC_S_LGATr_training_NoPID_Delphes_PU_PFfix_700_07_10_16_64_0.8_2025_05_24_23_01_07_703": "LGATr_GP_IRC_SN_700_07",
    }

    train_name = config["load_from_run"]
    ckpt_step = config["ckpt_step"]
    print("train name", train_name)
    if train_name not in training_datasets:
        print("!! unknown run", train_name)
    training_dataset = training_datasets.get(train_name, train_name) + "_s" + str(ckpt_step) + clust_suffix
    if "plptfilt01" in run_name.lower():
        training_dataset += "_PLPtFiltMinPt01" # min pt 0.1
    elif "noplfilter" in run_name.lower():
        training_dataset += "_noPLFilter"
    elif "noplptfilter" in run_name.lower():
        training_dataset += "_noPLPtFilter" # actually there was a 0.5 pt cut in the ntuplizer, removed by plptfilt01
    elif "nopletafilter" in run_name.lower():
        training_dataset += "_noPLEtaFilter"
    result["GT_R"] = gt_r
    result["training_dataset"] = training_dataset
    result["training_dataset_nostep"] = training_datasets.get(train_name, train_name)  + clust_suffix
    result["ckpt_step"] = ckpt_step
    return f"GT_R={gt_r} {training_dataset}, {prefix}", result


def flatten_list(lst):# lst is like [[0,0],[1,1]...]
    #return [item for sublist in lst for item in sublist]
    return list(chain.from_iterable(lst))

sz = 5
ak_path = os.path.join(path, "AKX", "count_matched_quarks")

result_PR_AKX = pickle.load(open(os.path.join(ak_path, "result_PR_AKX.pkl"), "rb"))
result_jet_props_akx = pickle.load(open(os.path.join(ak_path, "result_jet_properties_AKX.pkl"), "rb"))
result_qj_akx = pickle.load(open(os.path.join(ak_path, "result_quark_to_jet.pkl"), "rb"))
result_dq_pt_akx = pickle.load(open(os.path.join(ak_path, "result_pt_dq.pkl"), "rb"))
result_dq_mc_pt_akx = pickle.load(open(os.path.join(ak_path, "result_pt_mc_gt.pkl"), "rb"))
result_dq_props_akx = pickle.load(open(os.path.join(ak_path, "result_props_dq.pkl"), "rb"))

try:
    result_PR_AKX_PL = pickle.load(open(os.path.join(os.path.join(path, "AKX_PL", "count_matched_quarks"), "result_PR_AKX.pkl"), "rb"))
    result_qj_akx_PL = pickle.load(open(os.path.join(os.path.join(path, "AKX_PL", "count_matched_quarks"), "result_quark_to_jet.pkl"), "rb"))
    result_dq_mc_pt_akx_PL = pickle.load(open(os.path.join(os.path.join(path, "AKX_PL", "count_matched_quarks"), "result_pt_mc_gt.pkl"), "rb"))
    result_dq_pt_akx_PL = pickle.load(open(os.path.join(os.path.join(path, "AKX_PL", "count_matched_quarks"), "result_pt_dq.pkl"), "rb"))
    result_dq_props_akx_PL = pickle.load(open(os.path.join(os.path.join(path, "AKX_PL", "count_matched_quarks"), "result_props_dq.pkl"), "rb"))
except FileNotFoundError:
    print("FileNotFoundError")
    result_PR_AKX_PL = result_PR_AKX
    result_qj_akx_PL = result_qj_akx
try:
    result_PR_AKX_GL = pickle.load(open(os.path.join(os.path.join(path, "AKX_GL", "count_matched_quarks"), "result_PR_AKX.pkl"), "rb"))
    result_qj_akx_GL = pickle.load(open(os.path.join(os.path.join(path, "AKX_GL", "count_matched_quarks"), "result_quark_to_jet.pkl"), "rb"))
    result_dq_mc_pt_akx_GL = pickle.load(
        open(os.path.join(os.path.join(path, "AKX_GL", "count_matched_quarks"), "result_pt_mc_gt.pkl"), "rb"))
    result_dq_pt_akx_GL = pickle.load(
        open(os.path.join(os.path.join(path, "AKX_GL", "count_matched_quarks"), "result_pt_dq.pkl"), "rb"))
    result_dq_props_akx_GL = pickle.load(
        open(os.path.join(os.path.join(path, "AKX_GL", "count_matched_quarks"), "result_props_dq.pkl"), "rb"))

except FileNotFoundError:
    print("FileNotFoundError")
    result_PR_AKX_GL = result_PR_AKX

#plot_only = ["LGATr_GP", "LGATr_GP_IRC_S", "LGATr_GP_IRC_SN", "LGATr_GP_50k", "LGATr_GP_IRC_S_50k"]
plot_only = []

radius = [0.8]
def select_radius(d, radius, depth=3):
    # from the dictionary, select radius at the level
    if depth == 0:
        return d[radius]
    return {key: select_radius(d[key], radius, depth - 1) for key in d}

if len(models): # temporarily do not plot this one
    #fig, ax = plt.subplots(3, len(plot_only) + len(radius)*2, figsize=(sz * (len(plot_only)+len(radius)*2), sz * 3))
    # three columns: PL, GL, scouting for each model
    for i, model in tqdm(enumerate(sorted(models))):
        output_path = os.path.join(path, model, "count_matched_quarks")
        if not os.path.exists(os.path.join(output_path, "result.pkl")):
            print("Result not exists for model", model)
            continue
        result = pickle.load(open(os.path.join(output_path, "result.pkl"), "rb"))
        #result_unmatched = pickle.load(open(os.path.join(output_path, "result_unmatched.pkl"), "rb"))
        #result_fakes = pickle.load(open(os.path.join(output_path, "result_fakes.pkl"), "rb"))
        result_bc = pickle.load(open(os.path.join(output_path, "result_bc.pkl"), "rb"))
        result_PR = pickle.load(open(os.path.join(output_path, "result_PR.pkl"), "rb"))
        #matrix_plot(result, "Blues", "Avg. matched dark quarks / event").savefig(os.path.join(output_path, "avg_matched_dark_quarks.pdf"), ax=ax[0, i])
        #matrix_plot(result_fakes, "Greens", "Avg. unmatched jets / event").savefig(os.path.join(output_path, "avg_unmatched_jets.pdf"), ax=ax[1, i])
        #matrix_plot(result_PR, "Oranges", "Precision (N matched dark quarks / N predicted jets)", metric_comp_func = lambda r: r[0], ax=ax[0, i])
        #matrix_plot(result_PR, "Reds", "Recall (N matched dark quarks / N dark quarks)", metric_comp_func = lambda r: r[1], ax=ax[1, i])
        #matrix_plot(result_PR, "Purples", r"$F_1$ score", metric_comp_func = lambda r: 2 * r[0] * r[1] / (r[0] + r[1]), ax=ax[2, i])
        print("Getting run config for model", model)
        run_config_title, run_config = get_run_config(model)
        print("RC title", run_config_title)
        if run_config is None:
            print("Skipping", model)
            continue
        #ax[0, i].set_title(run_config_title)
        #ax[1, i].set_title(run_config_title)
        #ax[2, i].set_title(run_config_title)
        li = run_config["level_idx"]
        #ax_f1[i, li].set_title(run_config_title)
        #matrix_plot(result_PR, "Purples", r"$F_1$ score", metric_comp_func = lambda r: 2 * r[0] * r[1] / (r[0] + r[1]), ax=ax_f1[i, li])
        figures_all[run_config_title] = result_PR
        print(model, run_config_title)
        td, gtr, level, tdns = run_config["training_dataset"], run_config["GT_R"], run_config["level_idx"], run_config["training_dataset_nostep"]
        if tdns in plot_only or not len(plot_only):
            td = "R=" + str(gtr) + " " + td
            if td not in figures_all_sorted:
                figures_all_sorted[td] = {}
            figures_all_sorted[td][level] = figures_all[run_config_title]

    result_AKX_current = select_radius(result_PR_AKX, 0.8)
    result_AKX_PL = select_radius(result_PR_AKX_PL, 0.8)
    result_AKX_GL = select_radius(result_PR_AKX_GL, 0.8)

    figures_all_sorted["AK8"]: {
        0: result_AKX_PL,
        1: result_AKX_current,
        2: result_AKX_GL
    }
    for i, R in enumerate(radius):
        result_PR_AKX_current = select_radius(result_PR_AKX, R)
        #matrix_plot(result_PR_AKX_current, "Oranges", "Precision (N matched dark quarks / N predicted jets)",
        #            metric_comp_func=lambda r: r[0], ax=ax[0, i+len(models)])
        #matrix_plot(result_PR_AKX_current, "Reds", "Recall (N matched dark quarks / N dark quarks)",
        #            metric_comp_func=lambda r: r[1], ax=ax[1, i+len(models)])
        #matrix_plot(result_PR_AKX_current, "Purples", r"$F_1$ score", metric_comp_func=lambda r: 2 * r[0] * r[1] / (r[0] + r[1]),
        #            ax=ax[2, i+len(models)])
        #ax[0, i+len(models)].set_title(f"AK, R={R}")
        #ax[1, i+len(models)].set_title(f"AK, R={R}")
        #ax[2, i+len(models)].set_title(f"AK, R={R}")
        t = f"AK, R={R}"
        figures_all[t] = result_PR_AKX_current
    for i, R in enumerate(radius):
        result_PR_AKX_current = select_radius(result_PR_AKX_PL, R)
        #matrix_plot(result_PR_AKX_current, "Oranges", "Precision (N matched dark quarks / N predicted jets)",
        #            metric_comp_func=lambda r: r[0], ax=ax[0, i+len(models)+len(radius)])
        #matrix_plot(result_PR_AKX_current, "Reds", "Recall (N matched dark quarks / N dark quarks)",
        #            metric_comp_func=lambda r: r[1], ax=ax[1, i+len(models)+len(radius)])
        #matrix_plot(result_PR_AKX_current, "Purples", r"$F_1$ score", metric_comp_func=lambda r: 2 * r[0] * r[1] / (r[0] + r[1]),
        #            ax=ax[2, i+len(models)+len(radius)])
        #ax[0, i+len(models)+len(radius)].set_title(f"AK PL, R={R}")
        #ax[1, i+len(models)+len(radius)].set_title(f"AK PL, R={R}")
        #ax[2, i+len(models)+len(radius)].set_title(f"AK PL, R={R}")
        figures_all[f"AK PL, R={R}"] = result_PR_AKX_current
    for i, R in enumerate(radius):
        result_PR_AKX_current = select_radius(result_PR_AKX_GL, R)
        figures_all[f"AK GL, R={R}"] = result_PR_AKX_current
    #fig.tight_layout()
    #fig.savefig(out_file_PR)
    #print("Saved to", out_file_PR)
    #fig_f1.tight_layout().463
    #fig_f1.savefig(out_file_PRf1)
    pickle.dump(figures_all, open(out_file_PR.replace(".pdf", ".pkl"), "wb"))

figures_all_sorted["AK8"] = {
    0: select_radius(result_PR_AKX_PL, 0.8),
    1: select_radius(result_PR_AKX, 0.8),
    2: select_radius(result_PR_AKX_GL, 0.8)
}
text_level = ["PL", "PFCands", "GL"]

fig_f1, ax_f1 = plt.subplots(len(figures_all_sorted), 3, figsize=(sz * 2.5, sz * len(figures_all_sorted)))
if len(figures_all_sorted) == 1:
    ax_f1 = np.array([ax_f1])
for i in range(len(figures_all_sorted)):
    model = list(figures_all_sorted.keys())[i]
    renames = {
        "R=0.8 base_LGATr_s50000": "LGATr",
        "R=0.8 LGATr_GP_50k_s25020": "LGATr_GP",
        "R=0.8 LGATr_GP_IRC_S_50k_s12900": "LGATr_GP_IRC_S",
        "AK8": "AK8",
        "R=0.8 LGATr_GP_IRC_SN_50k_s22020": "LGATr_GP_IRC_SN"
    }
    for j in range(3):
        if j in figures_all_sorted[model]:
            if j in figures_all_sorted[model]:
                matrix_plot(figures_all_sorted[model][j], "Purples", r"$F_1$ score",
                            metric_comp_func=lambda r: 2 * r[0] * r[1] / (r[0] + r[1]), ax=ax_f1[i, j], is_qcd="qcd" in path.lower())
                ax_f1[i, j].set_title(renames.get(model, model) + " "+ text_level[j])
                ax_f1[i, j].set_xlabel("$m_{Z'}$")
                ax_f1[i, j].set_ylabel("$r_{inv.}$")
fig_f1.tight_layout()
fig_f1.savefig(out_file_PRf1)


import pandas as pd

# plot QCD results:
def get_qcd_results(i):
    # i=0: precision, i=1: recall, i=2: f1 score
    qcd_results = {}
    for model in figures_all_sorted:
        qcd_results[model] = {}
        for level in figures_all_sorted[model]:
            r = figures_all_sorted[model][level][0][0][0]
            r = [float(x) for x in r] # append f1 score
            r.append(r[0]*2*r[1] / (r[0]+r[1]))
            qcd_results[model][text_level[level]] = r[i]
    return pd.DataFrame(qcd_results).T

if "qcd" in path.lower():
    print("Precision:")
    print(get_qcd_results(0))
    print("----------------")
    print("Recall:")
    print(get_qcd_results(1))
    print("----------------")
    print("F1 score:")
    print(get_qcd_results(2))
## Now do the GT R vs metrics plots

oranges = plt.get_cmap("Oranges")
reds = plt.get_cmap("Reds")
purples = plt.get_cmap("Purples")

mDark = 20
if "qcd" in path.lower():
    print("QCD events")
    mDark=0
to_plot = {} # training dataset -> rInv -> mMed -> level -> "f1score" -> value
to_plot_steps = {} # training dataset -> rInv -> mMed -> level -> step -> value
to_plot_v2 = {} # level -> rInv -> mMed -> {"model": [P,R]}
quark_to_jet = {} # level -> rInv -> mMed -> model -> quark to jet assignment list

mc_gt_pt_of_dq = {}
pt_of_dq = {}
props_of_dq = {"eta": {}, "phi": {}} # Properties of dark quarks: eta and phi

results_all = {}
results_all_ak = {}
jet_properties = {} # training dataset -> rInv -> mMed -> level -> step -> jet property dict
jet_properties_ak = {} # rInv -> mMed -> level -> radius
plotting_hypotheses = [[700, 0.7], [700, 0.5], [700, 0.3], [900, 0.3], [900, 0.7]]
if "qcd" in path.lower():
    plotting_hypotheses = [[0,0]]
sz_small = 5
for j, model in enumerate(models):
    _, rc = get_run_config(model)
    if rc is None or model in ["Eval_eval_19March2025_pt1e-2_500particles_FT_PL_2025_04_02_14_28_33_421FT", "Eval_eval_19March2025_pt1e-2_500particles_FT_PL_2025_04_02_14_47_23_671FT", "Eval_eval_19March2025_small_aug_vanishing_momentum_2025_03_28_11_45_16_582", "Eval_eval_19March2025_small_aug_vanishing_momentum_2025_03_28_11_46_26_326"]:
        print("Skipping", model)
        continue
    td = rc["training_dataset"]
    td_raw = rc["training_dataset_nostep"]
    level = rc["level"]
    r = rc["GT_R"]
    output_path = os.path.join(path, model, "count_matched_quarks")
    if not os.path.exists(os.path.join(output_path, "result_PR.pkl")):
        continue
    result_PR = pickle.load(open(os.path.join(output_path, "result_PR.pkl"), "rb"))
    result_QJ = pickle.load(open(os.path.join(output_path, "result_quark_to_jet.pkl"), "rb"))
    result_jet_props = pickle.load(open(os.path.join(output_path, "result_jet_properties.pkl"), "rb"))
    result_MC_PT = pickle.load(open(os.path.join(output_path, "result_pt_mc_gt.pkl"), "rb"))
    result_PT_DQ = pickle.load(open(os.path.join(output_path, "result_pt_dq.pkl"), "rb"))
    result_DQ_props = pickle.load(open(os.path.join(output_path, "result_props_dq.pkl"), "rb"))
    print(level)
    if td not in to_plot:
        to_plot[td] = {}
        results_all[td] = {}
    if td_raw not in to_plot_steps:
        to_plot_steps[td_raw] = {}
        jet_properties[td_raw] = {}
    level_idx = ["PL", "scouting", "GL"].index(level)
    if level_idx not in to_plot_v2:
        to_plot_v2[level_idx] = {}
        quark_to_jet[level_idx] = {}
        pt_of_dq[level_idx] = {}
        mc_gt_pt_of_dq[level_idx] = {}
        for prop in props_of_dq:
            props_of_dq[prop][level_idx] = {}
    for mMed_h in result_PR:
        if mMed_h not in to_plot_v2[level_idx]:
            to_plot_v2[level_idx][mMed_h] = {}
            quark_to_jet[level_idx][mMed_h] = {}
            pt_of_dq[level_idx][mMed_h] = {}
            mc_gt_pt_of_dq[level_idx][mMed_h] = {}
            for prop in props_of_dq:
                props_of_dq[prop][level_idx][mMed_h] = {}
        if mMed_h not in to_plot_steps[td_raw]:
            to_plot_steps[td_raw][mMed_h] = {}
            jet_properties[td_raw][mMed_h] = {}
        if mMed_h not in results_all[td]:
            results_all[td][mMed_h] = {mDark: {}}
        for rInv_h in result_PR[mMed_h][mDark]:
            if rInv_h not in to_plot_v2[level_idx][mMed_h]:
                to_plot_v2[level_idx][mMed_h][rInv_h] = {}
                quark_to_jet[level_idx][mMed_h][rInv_h] = {}
                pt_of_dq[level_idx][mMed_h][rInv_h] = {}
                mc_gt_pt_of_dq[level_idx][mMed_h][rInv_h] = {}
                for prop in props_of_dq:
                    props_of_dq[prop][level_idx][mMed_h][rInv_h] = {}
            if rInv_h not in to_plot_steps[td_raw][mMed_h]:
                to_plot_steps[td_raw][mMed_h][rInv_h] = {}
                jet_properties[td_raw][mMed_h][rInv_h] = {}
            if level not in to_plot_steps[td_raw][mMed_h][rInv_h]:
                to_plot_steps[td_raw][mMed_h][rInv_h][level] = {}
                jet_properties[td_raw][mMed_h][rInv_h][level] = {}
            if rInv_h not in results_all[td][mMed_h][mDark]:
                results_all[td][mMed_h][mDark][rInv_h] = {}
            #for level in ["PL+ghosts", "GL+ghosts", "scouting+ghosts"]:
            if level not in results_all[td][mMed_h][mDark][rInv_h]:
                results_all[td][mMed_h][mDark][rInv_h][level] = {}
            precision = result_PR[mMed_h][mDark][rInv_h][0]
            recall = result_PR[mMed_h][mDark][rInv_h][1]
            f1score = 2 * precision * recall / (precision + recall)
            to_plot_v2[level_idx][mMed_h][rInv_h][td_raw] = [precision, recall]
            quark_to_jet[level_idx][mMed_h][rInv_h][td_raw] = result_QJ[mMed_h][mDark][rInv_h]
            pt_of_dq[level_idx][mMed_h][rInv_h][td_raw] = flatten_list(result_PT_DQ[mMed_h][mDark][rInv_h])
            mc_gt_pt_of_dq[level_idx][mMed_h][rInv_h][td_raw] = flatten_list(result_MC_PT[mMed_h][mDark][rInv_h])
            for prop in props_of_dq:
                props_of_dq[prop][level_idx][mMed_h][rInv_h][td_raw] = flatten_list(result_DQ_props[prop][mMed_h][mDark][rInv_h])
            #print("qj", quark_to_jet[level_idx][mMed_h][rInv_h][td_raw])
            if r not in results_all[td][mMed_h][mDark][rInv_h][level]:
                results_all[td][mMed_h][mDark][rInv_h][level][r] = f1score
            ckpt_step = rc["ckpt_step"]
            to_plot_steps[td_raw][mMed_h][rInv_h][level][ckpt_step] = f1score
            jet_properties[td_raw][mMed_h][rInv_h][level][ckpt_step] = result_jet_props[mMed_h][mDark][rInv_h]
m_Meds = []
r_invs = []
for key in to_plot_steps:
    m_Meds += list(to_plot_steps[key].keys())
    for key2 in to_plot_steps[key]:
        r_invs += list(to_plot_steps[key][key2].keys())

m_Meds = sorted(list(set(m_Meds)))
r_invs = sorted(list(set(r_invs)))

result_AKX_current = select_radius(result_PR_AKX, 0.8)
result_AKX_PL = select_radius(result_PR_AKX_PL, 0.8)
result_AKX_GL = select_radius(result_PR_AKX_GL, 0.8)
result_AKX_jet_properties = select_radius(result_jet_props_akx, 0.8)

jet_properties["AK8"] = {}
result_AKX_current_QJ = select_radius(result_qj_akx, 0.8)
result_AKX_PL_QJ = select_radius(result_qj_akx_PL, 0.8)
result_AKX_GL_QJ = select_radius(result_qj_akx_GL, 0.8)

result_AKX_current_pt_dq = select_radius(result_dq_pt_akx, 0.8)
result_AKX_PL_pt_dq = select_radius(result_dq_pt_akx_PL, 0.8)
result_AKX_GL_pt_dq = select_radius(result_dq_pt_akx_GL, 0.8)

result_AKX_current_MCpt_dq = select_radius(result_dq_mc_pt_akx, 0.8)
result_AKX_PL_MCpt_dq = select_radius(result_dq_mc_pt_akx_PL, 0.8)
result_AKX_GL_MCpt_dq = select_radius(result_dq_mc_pt_akx_GL, 0.8)

result_AKX_current_props_dq = select_radius(result_dq_props_akx, 0.8, depth=4)
result_AKX_PL_props_dq = select_radius(result_dq_props_akx_PL, 0.8, depth=4)
result_AKX_GL_props_dq = select_radius(result_dq_props_akx_GL, 0.8, depth=4)

from tqdm import tqdm

for mMed_h in result_AKX_jet_properties:
    for rInv_h in result_AKX_jet_properties[mMed_h][mDark]:
        if 0 in to_plot_v2:
            to_plot_v2[0][mMed_h][rInv_h]["AK8"] = result_AKX_PL[mMed_h][mDark][rInv_h]
            to_plot_v2[1][mMed_h][rInv_h]["AK8"] = result_AKX_current[mMed_h][mDark][rInv_h]
            to_plot_v2[2][mMed_h][rInv_h]["AK8"] = result_AKX_GL[mMed_h][mDark][rInv_h]
            quark_to_jet[0][mMed_h][rInv_h]["AK8"] = result_AKX_PL_QJ[mMed_h][mDark][rInv_h]
            quark_to_jet[1][mMed_h][rInv_h]["AK8"] = result_AKX_current_QJ[mMed_h][mDark][rInv_h]
            quark_to_jet[2][mMed_h][rInv_h]["AK8"] = result_AKX_GL_QJ[mMed_h][mDark][rInv_h]
            pt_of_dq[0][mMed_h][rInv_h]["AK8"] = flatten_list(result_AKX_PL_pt_dq[mMed_h][mDark][rInv_h])
            pt_of_dq[1][mMed_h][rInv_h]["AK8"] = flatten_list(result_AKX_current_pt_dq[mMed_h][mDark][rInv_h])
            pt_of_dq[2][mMed_h][rInv_h]["AK8"] = flatten_list(result_AKX_GL_pt_dq[mMed_h][mDark][rInv_h])
            mc_gt_pt_of_dq[0][mMed_h][rInv_h]["AK8"] = flatten_list(result_AKX_PL_MCpt_dq[mMed_h][mDark][rInv_h])
            mc_gt_pt_of_dq[1][mMed_h][rInv_h]["AK8"] = flatten_list(result_AKX_current_MCpt_dq[mMed_h][mDark][rInv_h])
            mc_gt_pt_of_dq[2][mMed_h][rInv_h]["AK8"] = flatten_list(result_AKX_GL_MCpt_dq[mMed_h][mDark][rInv_h])
            for k in props_of_dq:
                props_of_dq[k][0][mMed_h][rInv_h]["AK8"] = flatten_list(result_AKX_PL_props_dq[k][mMed_h][mDark][rInv_h])
                props_of_dq[k][1][mMed_h][rInv_h]["AK8"] = flatten_list(result_AKX_current_props_dq[k][mMed_h][mDark][rInv_h])
                props_of_dq[k][2][mMed_h][rInv_h]["AK8"] = flatten_list(result_AKX_GL_props_dq[k][mMed_h][mDark][rInv_h])
        if mMed_h not in jet_properties["AK8"]:
            jet_properties["AK8"][mMed_h] = {}
        if rInv_h not in jet_properties["AK8"][mMed_h]:
            jet_properties["AK8"][mMed_h][rInv_h] = {}
        jet_properties["AK8"][mMed_h][rInv_h] = {"scouting": {50000: result_AKX_jet_properties[mMed_h][mDark][rInv_h]}}

rename_results_dict = {
    "LGATr_comparison_DifferentTrainingDS": "base",
    "LGATr_comparison_GP_training": "GP",
    "LGATr_comparison_GP_IRC_S_training": "GP_IRC_S",
    "LGATr_comparison_GP_IRC_SN_training": "GP_IRC_SN"
}


hypotheses_to_plot = [[0,0],[700,0.7],[700,0.5],[700,0.3]]



def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def get_label_from_superset(lbl, labels_rename, labels):
    if lbl == '':
        return "Missed by all"
    r = [labels[int(i)] for i in lbl]
    r = [labels_rename.get(l,l) for l in r]
    if len(r) == 2 and "QCD" in r and "900_03" in r:
        return "Found by both models but not AK"
    if len(r) == 3:
        return "Found by all"
    return ", ".join(r)


for hyp_m, hyp_rinv in hypotheses_to_plot:
    if 0 not in to_plot_v2:
        continue # Not for the lower-pt thresholds, where only GL and PL are available
    if hyp_m not in to_plot_v2[0] or hyp_rinv not in to_plot_v2[0][hyp_m]:
        continue
    # plot here the venn diagrams
    labels = ["LGATr_GP_IRC_S_QCD", "AK8", "LGATr_GP_IRC_S_50k"]
    labels_global = ["LGATr_GP_IRC_S_QCD", "AK8", "LGATr_GP_IRC_S_50k"]
    labels_rename = {"LGATr_GP_IRC_S_QCD": "QCD", "LGATr_GP_IRC_S_50k": "900_03"}
    fig_venn, ax_venn = plt.subplots(6, 3, figsize=(5 * 3, 5 * 6)) # the bottom ones are for pt of the DQ, pt of the MC GT, pt of MC GT / pt of DQ, eta, and phi distributions
    fig_venn1, ax_venn1 = plt.subplots(6, 2, figsize=(5*2, 5*6)) # Only the PFCands-level, with full histogram on the left and density on the right
    for level in range(3):
        #labels = list(results_dict["LGATr_comparison_GP_IRC_S_training"][0].keys())
        label_combination_to_number = {} # fill it with all possible label combinations e.g. if there are 3 labels: "NA", "0", "1", "2", "01", "012", "12", "02"
        powerset_str = ["".join([str(x) for x in sorted(list(a))]) for a in powerset(range(len(labels)))]
        set_to_count = {key: 0 for key in powerset_str}
        set_to_stats = {key: {"pt_dq": [], "pt_mc_t": [], "pt_mc_t_dq_ratio": [], "eta": [], "phi": []} for key in powerset_str}
        label_to_result = {}
        #label_to_stats = {"pt_dq": , "pt_mc_t": [], "pt_mc_t_dq_ratio": [], "eta": [], "phi": []}
        n_dq = 999999999
        for j, label in enumerate(labels):
            r = flatten_list(quark_to_jet[level][hyp_m][hyp_rinv][label])
            n_dq = min(n_dq, len(r)) # Find the minimum number of dark quarks in all labels
        for j, label in enumerate(labels):
            r = torch.tensor(flatten_list(quark_to_jet[level][hyp_m][hyp_rinv][label]))
            r = (r != -1) # Whether quark no. X is caught or not
            label_to_result[j] = r.tolist()[:n_dq]
            #r = torch.tensor(flatten_list(pt_of_dq[level][hyp_m][hyp_rinv][label]))
            #r = r[:n_dq]
            #label_to_stats["pt_dq"].append(r.tolist())
            #r1 = torch.tensor(flatten_list(mc_gt_pt_of_dq[level][hyp_m][hyp_rinv][label]))
            #r1 = r1[:n_dq]
            #label_to_stats["pt_mc_t"].append(r1.tolist())
            #r2 = r1 / r
            #r2 = r2[:n_dq]
            #label_to_stats["pt_mc_t_dq_ratio"].append(r2.tolist())
            #r_eta = torch.tensor(flatten_list(props_of_dq["eta"][level][hyp_m][hyp_rinv][label]))
            #r_eta = r_eta[:n_dq]
            #label_to_stats["eta"].append(r_eta.tolist())
            ##r_phi = torch.tensor(flatten_list(props_of_dq["phi"][level][hyp_m][hyp_rinv][label]))
            #r_phi = r_phi[:n_dq]
            #label_to_stats["phi"].append(r_phi.tolist())
            assert len(label_to_result[j]) == n_dq, f"Label {label} has different number of quarks than others {n_dq} != {len(label_to_result[j])}"
            #n_dq = min(n_dq, len(r))
        #for j, label in enumerate(labels):
        #    assert len(label_to_result[j]) == n_dq, f"Label {label} has different number of quarks than others {n_dq} != {len(label_to_result[j])}"
        for c in tqdm(range(n_dq)):
            belonging_to_set = ""
            for j, label in enumerate(labels):
                if label_to_result[j][c] == 1:
                    belonging_to_set += str(j)
            set_to_count[belonging_to_set] += 1
            #for key in label_to_stats:
            #    for idx in belonging_to_set:
            #        idx_int = int(idx) # e.g. "0", "1" etc.
            #        set_to_stats[belonging_to_set]
            for j, label in enumerate(labels):
                current_dq_pt = pt_of_dq[level][hyp_m][hyp_rinv][label][c]
                current_mc_gt_pt = mc_gt_pt_of_dq[level][hyp_m][hyp_rinv][label][c]
                current_dq_eta = props_of_dq["eta"][level][hyp_m][hyp_rinv][label][c]
                current_dq_phi = props_of_dq["phi"][level][hyp_m][hyp_rinv][label][c]
                set_to_stats[belonging_to_set]["pt_dq"].append(current_dq_pt)
                set_to_stats[belonging_to_set]["pt_mc_t"].append(current_mc_gt_pt)
                set_to_stats[belonging_to_set]["pt_mc_t_dq_ratio"].append(current_mc_gt_pt/current_dq_pt)
                set_to_stats[belonging_to_set]["eta"].append(current_dq_eta)
                set_to_stats[belonging_to_set]["phi"].append(current_dq_phi)
        #print("set_to_count for level", level, ":", set_to_count, "labels:", labels)
        title = f"$m_{{Z'}}={hyp_m}$ GeV, $r_{{inv.}}={hyp_rinv}$, {text_level[level]} (missed by all: {set_to_count['']}) "
        if hyp_m == 0 and hyp_rinv == 0:
            title = f"QCD, {text_level[level]} (missed by all: {set_to_count['']})"
        ax_venn[0, level].set_title(title)
        plot_venn3_from_index_dict(ax_venn[0, level], set_to_count, set_labels=[labels_rename.get(l,l) for l in labels], set_colors=["orange", "gray", "red"])
        if level == 1: #reco-level
            plot_venn3_from_index_dict(ax_venn1[0, 1], set_to_count,
                                   set_labels=[labels_rename.get(l,l) for l in labels],
                                   set_colors=["orange", "gray", "red"])
        bins = {
            "pt_dq": np.linspace(90, 250, 50),
            "pt_mc_t": np.linspace(0, 200, 50),
            "pt_mc_t_dq_ratio": np.linspace(0, 1.3, 30),
            "eta": np.linspace(-4, 4, 20),
            "phi": np.linspace(-np.pi, np.pi, 20)
        }
        # 10 random colors
        clrs = ["green", "red", "orange", "pink", "blue", "purple", "cyan", "magenta"]
        key_rename_dict = {"pt_dq": "$p_T$ of quark", "pt_mc_t": "$p_T$ of particles within radius of R=0.8 of quark", "pt_mc_t_dq_ratio": "$p_T$ (part. within R=0.8 of quark) / $p_T$ (quark) ", "eta": "$\eta$ of quark", "phi": "$\phi$ of quark" }
        for k, key in enumerate(["pt_dq", "pt_mc_t", "pt_mc_t_dq_ratio", "eta", "phi"]):
            for s_idx, s in enumerate(sorted(set_to_stats.keys())):
                if len(set_to_stats[s][key]) == 0:
                    continue
                lbl = s
                #if s == "":
                #    lbl = "none"
                lbl1 = get_label_from_superset(lbl, labels_rename, labels)

                if lbl1 not in ["Missed by all", "Found by both models but not AK", "AK8", "Found by all"]:
                    continue
                if level == 1:
                    ax_venn1[k + 1, 1].hist(set_to_stats[s][key], bins=bins[key], histtype="step",
                                           label=lbl1, color=clrs[s_idx], density=True)
                    ax_venn1[k + 1, 0].set_title(f"{key_rename_dict[key]}")
                    ax_venn1[k+1, 1].set_title(f"{key_rename_dict[key]}")
                    ax_venn1[k + 1, 1].set_ylabel("Density")
                if lbl not in ["", "012"]:
                    # We are only interested in the differences...
                    ax_venn[k+1, level].hist(set_to_stats[s][key], bins=bins[key], histtype="step", label=lbl1, color=clrs[s_idx])
                    ax_venn[k+1, level].set_title(f"{key_rename_dict[key]}")
                    if level == 1:
                        ax_venn1[k + 1, 0].hist(set_to_stats[s][key], bins=bins[key], histtype="step",
                                            label=lbl1,
                                            color=clrs[s_idx])
                #ax_venn[k+1, level].set_xlabel(key)
                #ax_venn[k+1, level].set_ylabel("Count")
        for k in range(5):
            ax_venn[k+1, level].legend()
    fig_venn.tight_layout()
    for k in range(5):
        ax_venn1[k+1, 0].legend()
        ax_venn1[k+1, 1].legend()
    fig_venn1.tight_layout()
    f = os.path.join(get_path(args.input, "results"), f"venn_diagram_{hyp_m}_{hyp_rinv}.pdf")
    fig_venn.savefig(f)
    f1 = os.path.join(get_path(args.input, "results"), f"venn_diagram_{hyp_m}_{hyp_rinv}_reco_level_only.pdf")
    fig_venn1.savefig(f1)

    for i, lbl in enumerate(["precision", "recall", "F1"]): # 0=precision, 1=recall, 2=F1
        sz_small1 = 2.5
        fig, ax = plt.subplots(len(rename_results_dict), 3, figsize=(sz_small1 * 3, sz_small1 * len(rename_results_dict)))
        for i1, key in enumerate(list(rename_results_dict.keys())):
            for level in range(3):
                level_text = text_level[level]
                labels = list(results_dict[key][0].keys())
                colors = [results_dict[key][0][l] for l in labels]
                res_precision = np.array([to_plot_v2[level][hyp_m][hyp_rinv][l][0] for l in labels])
                res_recall = np.array([to_plot_v2[level][hyp_m][hyp_rinv][l][1] for l in labels])
                res_f1 = 2 * res_precision * res_recall / (res_precision + res_recall)
                if i == 0:
                    values = res_precision
                elif i == 1:
                    values = res_recall
                else:
                    values = res_f1
                rename_dict = results_dict[key][1]
                labels_renamed = [rename_dict.get(l,l) for l in labels]
                print(i1, level)
                ax_tiny_histogram(ax[i1, level], labels_renamed, colors, values)
                ax[i1, level].set_title(f"{rename_results_dict[key]} {level_text}")
        fig.tight_layout()
        fig.savefig(os.path.join(get_path(args.input, "results"), f"{lbl}_results_by_level_{hyp_m}_{hyp_rinv}_{key}.pdf"))


for hyp_m, hyp_rinv in hypotheses_to_plot:
    if 0 not in to_plot_v2:
        continue # Not for the lower-pt thresholds, where only GL and PL are available
    if hyp_m not in to_plot_v2[0] or hyp_rinv not in to_plot_v2[0][hyp_m]:
        continue
    # plot here the venn diagrams
    labels = ["LGATr_GP_IRC_S_QCD", "AK8", "LGATr_GP_IRC_S_50k"]
    labels_global = ["LGATr_GP_IRC_S_QCD", "AK8", "LGATr_GP_IRC_S_50k"]
    labels_rename = {"LGATr_GP_IRC_S_QCD": "QCD", "LGATr_GP_IRC_S_50k": "900_03"}
    fig_venn2, ax_venn2 = plt.subplots(1, len(labels), figsize=(4*len(labels), 4)) # the bottom ones are for pt of the DQ, pt of the MC GT, pt of MC GT / pt of DQ, eta, and phi distributions
    for j, label in enumerate(labels):
        #labels = list(results_dict["LGATr_comparison_GP_IRC_S_training"][0].keys())
        label_combination_to_number = {} # fill it with all possible label combinations e.g. if there are 3 labels: "NA", "0", "1", "2", "01", "012", "12", "02"
        powerset_str = ["".join([str(x) for x in sorted(list(a))]) for a in powerset(range(3))]
        set_to_count = {key: 0 for key in powerset_str}
        label_to_result = {}
        n_dq = 99999999 # Sometimes, the last batch gets cut off etc. ...
        for level in range(3):
            r = flatten_list(quark_to_jet[level][hyp_m][hyp_rinv][label])
            n_dq = min(n_dq, len(r))
        for level in range(3):
            r = torch.tensor(flatten_list(quark_to_jet[level][hyp_m][hyp_rinv][label]))
            r = (r != -1)
            label_to_result[level] = r.tolist()[:n_dq]
            assert len(label_to_result[level]) == n_dq, f"Label {label} has different number of quarks than others {n_dq} != {len(label_to_result[level])}"
        for c in tqdm(range(n_dq)):
            belonging_to_set = ""
            for lvl in range(3):
                if label_to_result[lvl][c] == 1:
                    belonging_to_set += str(lvl)
            set_to_count[belonging_to_set] += 1
        if hyp_m == 0 and hyp_rinv == 0:
            title = f"QCD, {label} (missed by all: {set_to_count['']}) "
        else:
            title = f"$m_{{Z'}}={hyp_m}$ GeV, $r_{{inv.}}={hyp_rinv}$, {label} (miss: {set_to_count['']}) "
        ax_venn2[j].set_title(title)
        plot_venn3_from_index_dict(ax_venn2[j], set_to_count, set_labels=text_level, set_colors=["orange", "gray", "red"], remove_max=1)
    fig_venn2.tight_layout()
    f = os.path.join(get_path(args.input, "results"), f"venn_diagram_{hyp_m}_{hyp_rinv}_Agreement_between_levels.pdf")
    fig_venn2.savefig(f)


for key in results_dict:
    for level in range(3):
        level_text = text_level[level]
        labels = list(results_dict[key][0].keys())
        if level in to_plot_v2:
            f, a = multiple_matrix_plot(to_plot_v2[level], labels=labels, colors=[results_dict[key][0][l] for l in labels], rename_dict=results_dict[key][1])
            if f is None:
                print("No figure for", key, level)
                continue
            #f.suptitle(f"{level_text} $F_1$ score")
            out_file = f"grid_stack_F1_{level_text}_{key}.pdf"
            out_file = os.path.join(get_path(args.input, "results"), out_file)
            f.savefig(out_file)
            print("Saved to", out_file)

from matplotlib.lines import Line2D

# Define custom legend handles
custom_lines = [
    Line2D([0], [0], color='orange', linestyle='-', label='LGATr'),
    Line2D([0], [0], color='green', linestyle='-', label='GATr'),
    Line2D([0], [0], color='blue', linestyle='-', label='Transformer'),
    Line2D([0], [0], color='gray', linestyle='-', label='AK8'),
    Line2D([0], [0], color='black', linestyle='-', label='reco'),
    Line2D([0], [0], color='black', linestyle=':', label='gen'),
    Line2D([0], [0], color='black', linestyle='--', label='parton'),
]

if len(models):
    fig_steps, ax_steps = plt.subplots(len(m_Meds), len(r_invs),  figsize=(sz_small * len(r_invs), sz_small * len(m_Meds)))
    if len(m_Meds) == 1 and len(r_invs) == 1:
        ax_steps = np.array([[ax_steps]])
    histograms = {}

    for key in histograms_dict:
        if key not in histograms:
            histograms[key] = {}
        for i in ["pt", "eta", "phi"]:
            f, a = plt.subplots(len(m_Meds), len(r_invs), figsize=(sz_small * len(r_invs), sz_small * len(m_Meds)))
            if len(r_invs) == 1 and len(m_Meds) == 1:
                a = np.array([[a]])
            histograms[key][i] = f, a
    colors = {"base_LGATr": "orange", "base_Tr": "blue", "base_GATr": "green", "AK8": "gray"} # THE COLORS FOR THE STEP VS. F1 SCORE
    #colors_small_dataset = {"base_LGATr_SD": "orange", "base_Tr_SD": "blue", "base_GATr_SD": "green", "AK8": "gray"}
    #colors = colors_small_dataset
    level_styles = {"scouting": "solid", "PL": "dashed", "GL": "dotted"}
    #step_to_plot_histograms = 50000  # phi, eta, pt histograms...
    level_to_plot_histograms = "scouting"


    for i, mMed_h in enumerate(m_Meds):
        for j, rInv_h in enumerate(r_invs):
            ax_steps[i, j].set_title("$m_{{Z'}} = {}$ GeV, $r_{{inv.}} = {}$".format(mMed_h, rInv_h))
            ax_steps[i, j].set_xlabel("Training step")
            ax_steps[i, j].set_ylabel("Test $F_1$ score")
            #if j == 0:
                #ax_steps[i, j].set_ylabel("$m_{{Z'}} = {}$".format(mMed_h))
                #for subset in histograms:
                    #for key in histograms[subset]:
                        #histograms[subset][key][1][i, j].set_ylabel("$m_{{Z'}} = {}$".format(mMed_h))
            if i == len(m_Meds)-1:
                ax_steps[i, j].set_xlabel("$r_{{inv.}} = {}$".format(rInv_h))
                for subset in histograms:
                    for key in histograms[subset]:
                        histograms[subset][key][1][i, j].set_xlabel("$r_{{inv.}} = {}$".format(rInv_h))
            for model in jet_properties:
                if level_to_plot_histograms not in jet_properties[model][mMed_h][rInv_h]:
                    print("Skipping", model, level_to_plot_histograms, " - levels:", jet_properties[model][mMed_h][rInv_h].keys())
                    continue
                for subset in histograms:
                    for key in histograms[subset]:
                        if model not in histograms_dict[subset][1]:
                            continue
                        step_to_plot_histograms = histograms_dict[subset][0][model]
                        if step_to_plot_histograms not in jet_properties[model][mMed_h][rInv_h][level_to_plot_histograms]:
                            print("Swapping the step to plot histograms", jet_properties[model][mMed_h][rInv_h][level_to_plot_histograms].keys())
                            step_to_plot_histograms = sorted(list(jet_properties[model][mMed_h][rInv_h][level_to_plot_histograms].keys()))[0]
                        pred = np.array(jet_properties[model][mMed_h][rInv_h][level_to_plot_histograms][step_to_plot_histograms][key + "_pred"])
                        truth = np.array(jet_properties[model][mMed_h][rInv_h][level_to_plot_histograms][step_to_plot_histograms][key + "_gen_particle"])
                        if key.startswith("pt"):
                            q = pred/truth
                            symbol = "/" # division instead of subtraction symbol for pt
                            quantity = "p_{T,pred}/p_{T,true}"
                            bins = np.linspace(0, 2.5, 100)
                        elif key.startswith("eta"):
                            q = (pred - truth)
                            symbol = "-"
                            quantity="\eta_{pred}-\eta_{true}"
                            bins = np.linspace(-0.8, 0.8, 50)
                        elif key.startswith("phi"):
                            q = pred - truth
                            symbol = "-"
                            quantity = "\phi_{pred}-\phi_{true}"
                            q[q > np.pi] -= 2 * np.pi
                            q[q< -np.pi] += 2 * np.pi
                            bins = np.linspace(-0.8, 0.8, 50)
                            print("Max", np.max(q), "Min", np.min(q))
                        rename = {"base_LGATr": "LGATr",
                                  "LGATr_GP_IRC_S_50k": "LGATr_GP_IRC_S",
                                  "AK8": "AK8",
                                  "LGATr_GP_50k": "LGATr_GP"}
                        histograms[subset][key][1][i, j].hist(q, histtype="step", color=histograms_dict[subset][1][model], label=rename.get(model, model), bins=bins, density=True)
                        if mMed_h > 0:
                            histograms[subset][key][1][i, j].set_title(f"${quantity}$ $m_{{Z'}}={mMed_h}$ GeV, $r_{{inv.}}={rInv_h}$")
                        else:
                            histograms[subset][key][1][i, j].set_title(f"${quantity}$")
                        histograms[subset][key][1][i, j].legend()
                        histograms[subset][key][1][i, j].grid(True)
            for model in to_plot_steps:
                for lvl in to_plot_steps[model][mMed_h][rInv_h]:
                    if model not in colors:
                        print("Skipping", model)
                        continue
                    print(model)
                    ls = level_styles[lvl]
                    plt_dict = to_plot_steps[model][mMed_h][rInv_h][lvl]
                    x_pts = sorted(list(plt_dict.keys()))
                    y_pts = [plt_dict[k] for k in x_pts]
                    if ls == "solid":
                        ax_steps[i, j].plot(x_pts, y_pts, label=model, marker=".", linestyle=ls, color=colors[model])
                    else:
                        # No label
                        ax_steps[i, j].plot(x_pts, y_pts, marker=".", linestyle=ls, color=colors[model])
                    ax_steps[i, j].legend(handles=custom_lines)
                    # now plot a horizontal line for the AKX same level
                    if lvl == "scouting":
                        rc = result_AKX_current
                    elif lvl == "PL":
                        rc = result_AKX_PL
                    elif lvl == "GL":
                        rc = result_AKX_GL
                    else:
                        raise Exception
                    pr = rc[mMed_h][mDark][rInv_h][0]
                    rec = rc[mMed_h][mDark][rInv_h][1]
                    f1ak = 2 * pr * rec / (pr + rec)
                    ax_steps[i, j].axhline(f1ak, color="gray", linestyle=ls, alpha=0.5)
                ax_steps[i, j].grid(1)
    path_steps_fig = os.path.join(get_path(args.input, "results"), "score_vs_step_plots.pdf")
    fig_steps.tight_layout()
    fig_steps.savefig(path_steps_fig)
    for subset in histograms:
        for key in histograms[subset]:
            fig = histograms[subset][key][0]
            fig.tight_layout()
            fig.savefig(os.path.join(get_path(args.input, "results"), "histogram_{}_{}.pdf".format(key, subset)))
    print("Saved to", path_steps_fig)


'''for i, h in enumerate(plotting_hypotheses):
        mMed_h, rInv_h = h
        if rInv_h not in to_plot[td]:
            to_plot[td][rInv_h] = {}
        print("Model", model)
        if mMed_h not in to_plot[td][rInv_h]:
            to_plot[td][rInv_h][mMed_h] = {} # level
        if level not in to_plot[td][rInv_h][mMed_h]:
            to_plot[td][rInv_h][mMed_h][level] = {"precision": [], "recall": [], "f1score": [], "R": []}
        precision = result_PR[mMed_h][mDark][rInv_h][0]
        recall = result_PR[mMed_h][mDark][rInv_h][1]
        f1score = 2 * precision * recall / (precision + recall)
        to_plot[td][rInv_h][mMed_h][level]["precision"].append(precision)
        to_plot[td][rInv_h][mMed_h][level]["recall"].append(recall)
        to_plot[td][rInv_h][mMed_h][level]["f1score"].append(f1score)
        to_plot[td][rInv_h][mMed_h][level]["R"].append(r)

'''
to_plot_ak = {} # level ("scouting"/"GL"/"PL") -> rInv -> mMed -> {"f1score": [], "R": []}

for j, model in enumerate(["AKX", "AKX_PL", "AKX_GL"]):
    print(model)
    if os.path.exists(os.path.join(path, model, "count_matched_quarks", "result_PR_AKX.pkl")):
        result_PR_AKX = pickle.load(open(os.path.join(path, model, "count_matched_quarks", "result_PR_AKX.pkl"), "rb"))
    else:
        print("Skipping", model)
        continue
    level = "scouting"
    if "PL" in model:
        level = "PL"
    elif "GL" in model:
        level = "GL"
    if level not in to_plot_ak:
        to_plot_ak[level] = {}
    for mMed_h in result_PR_AKX:
        if mMed_h not in results_all_ak:
            results_all_ak[mMed_h] = {mDark: {}}
        for rInv_h in result_PR_AKX[mMed_h][mDark]:
            if rInv_h not in results_all_ak[mMed_h][mDark]:
                results_all_ak[mMed_h][mDark][rInv_h] = {}
            if level not in results_all_ak[mMed_h][mDark][rInv_h]:
                results_all_ak[mMed_h][mDark][rInv_h][level] = {}
            for ridx, R in enumerate(result_PR_AKX[mMed_h][mDark][rInv_h]):
                if R not in results_all_ak[mMed_h][mDark][rInv_h][level]:
                    precision = result_PR_AKX[mMed_h][mDark][rInv_h][R][0]
                    recall = result_PR_AKX[mMed_h][mDark][rInv_h][R][1]
                    f1score = 2 * precision * recall / (precision + recall)
                    results_all_ak[mMed_h][mDark][rInv_h][level][R] = f1score

    for i, h in enumerate(plotting_hypotheses):
        mMed_h, rInv_h = h
        if rInv_h not in to_plot_ak[level]:
            to_plot_ak[level][rInv_h] = {}
        print("Model", model)
        if mMed_h not in to_plot_ak[level][rInv_h]:
            to_plot_ak[level][rInv_h][mMed_h] = {"precision": [], "recall": [], "f1score": [], "R": []}
        rs = sorted(result_PR_AKX[mMed_h][mDark][rInv_h].keys())
        precision = np.array([result_PR_AKX[mMed_h][mDark][rInv_h][i][0] for i in rs])
        recall = np.array([result_PR_AKX[mMed_h][mDark][rInv_h][i][1] for i in rs])
        f1score = 2 * precision * recall / (precision + recall)
        to_plot_ak[level][rInv_h][mMed_h]["precision"] = precision
        to_plot_ak[level][rInv_h][mMed_h]["recall"] = recall
        to_plot_ak[level][rInv_h][mMed_h]["f1score"] = f1score
        to_plot_ak[level][rInv_h][mMed_h]["R"] = rs
print("AK:", to_plot_ak)
fig, ax = plt.subplots(len(to_plot) + 1, len(plotting_hypotheses), figsize=(sz_small * len(plotting_hypotheses), sz_small * len(to_plot))) # also add AKX as last plot

if len(to_plot) == 0:
    ax = np.array([ax])
colors = {
    #"PL": "green",
    #"GL": "blue",
    #"scouting": "red",
    "PL+ghosts": "green",
    "GL+ghosts": "blue",
    "scouting+ghosts": "red"
}
ak_colors = {
    "PL": "green",
    "GL": "blue",
    "scouting": "red",
}
'''
for i, td in enumerate(to_plot):
    # for each training dataset
    for j, h in enumerate(plotting_hypotheses):
        ax[i, j].set_title(f"r_inv={h[1]}, m={h[0]}, tr. on {td}")
        ax[i, j].set_ylabel("F1 score")
        ax[i, j].set_xlabel("GT R")
        ax[i, j].grid()
        for level in sorted(list(to_plot[td][h[1]][h[0]].keys())):
            print("level", level)
            print("Plotting", td, h[1], h[0], level)
            if level in colors:
                ax[i, j].plot(to_plot[td][h[1]][h[0]][level]["R"], to_plot[td][h[1]][h[0]][level]["f1score"], ".-", label=level, color=colors[level])
        ax[i, j].legend()
for j, h in enumerate(plotting_hypotheses): # for to_plot_AK
    ax[-1, j].set_title(f"r_inv={h[1]}, m={h[0]}, AK baseline")
    ax[-1, j].set_ylabel("F1 score")
    ax[-1, j].set_xlabel("GT R")
    ax[-1, j].grid()
    for i, ak_level in enumerate(sorted(list(to_plot_ak.keys()))):
        mMed_h, rInv_h = h
        if ak_level in ak_colors:
            ax[-1, j].plot(to_plot_ak[ak_level][rInv_h][mMed_h]["R"], to_plot_ak[ak_level][rInv_h][mMed_h]["f1score"], ".-", label=ak_level, color=ak_colors[ak_level])
    ax[-1, j].legend()
fig.tight_layout()
fig.savefig(os.path.join(get_path(args.input, "results"), "score_vs_GT_R_plots_1.pdf"))
print("Saved to", os.path.join(get_path(args.input, "results"), "score_vs_GT_R_plots_1.pdf"))
'''
fig, ax = plt.subplots(1, len(results_all)*len(radius) + len(radius), figsize=(7 * len(results_all)*len(radius)+len(radius), 5))
for i, model in enumerate(results_all):
    for j, R in enumerate(radius):
        #if r not in results_all[model][700][20][0.3]["scouting"]:
        #    continue
        # for each training dataset
        index = len(radius)*i + j
        ax[index].set_title(model + " R={}".format(R))
        matrix_plot(results_all[model], "Greens", r"PL/GL F1 score", ax=ax[index], metric_comp_func=lambda r: r["PL+ghosts"][R]/r["scouting+ghosts"][R])
for i, R in enumerate(radius):
    index = len(radius)*len(results_all) + i
    ax[index].set_title("AK R={}".format(R))
    matrix_plot(results_all_ak, "Greens", r"PL/GL F1 score", ax=ax[index], metric_comp_func=lambda r: r["PL"][R]/r["GL"][R])
fig.tight_layout()
fig.savefig(out_file_PG)

print("Saved to", out_file_PG)
1/0


#print("Saved to", os.path.join(get_path(args.input, "results"), "score_vs_GT_R_plots_AK.pdf"))
#print("Saved to", os.path.join(get_path(args.input, "results"), "score_vs_GT_R_plots_AK_ratio.pdf"))

########### Now save the above plot with objectness score applied

if args.threshold_obj_score != -1:
    fig, ax = plt.subplots(3, len(models), figsize=(sz * len(models), sz * 3))
    for i, model in tqdm(enumerate(models)):
        output_path = os.path.join(path, model, "count_matched_quarks")
        if not os.path.exists(os.path.join(output_path, "result.pkl")):
            continue
        result = pickle.load(open(os.path.join(output_path, "result.pkl"), "rb"))
        #result_unmatched = pickle.load(open(os.path.join(output_path, "result_unmatched.pkl"), "rb"))
        result_fakes = pickle.load(open(os.path.join(output_path, "result_fakes.pkl"), "rb"))
        result_bc = pickle.load(open(os.path.join(output_path, "result_bc.pkl"), "rb"))
        result_PR = pickle.load(open(os.path.join(output_path, "result_PR.pkl"), "rb"))
        result_PR_thresholds = pickle.load(open(os.path.join(output_path, "result_PR_thresholds.pkl"), "rb"))
        #thresholds = sorted(list(result_PR_thresholds[900][20][0.3].keys()))
        #thresholds = np.array(thresholds)
        # now linearly interpolate the thresholds and set the j according to args.threshold_obj_score
        j = np.argmin(np.abs(thresholds - args.threshold_obj_score))
        print("Thresholds", thresholds)
        print("Chose j=", j, "for threshold", args.threshold_obj_score, "(effectively it's", thresholds[j], ")")
        def wrap(r):
            # compute [precision, recall] array from [n_relevant_retrieved, all_retrieved, all_relevant]
            if r[1] == 0 or r[2] == 0:
                return [0, 0]
            return [r[0] / r[1], r[0] / r[2]]
        matrix_plot(result_PR_thresholds, "Oranges", "Precision (N matched dark quarks / N predicted jets)", metric_comp_func = lambda r: wrap(r[j])[0], ax=ax[0, i])
        matrix_plot(result_PR_thresholds, "Reds", "Recall (N matched dark quarks / N dark quarks)", metric_comp_func = lambda r: wrap(r[j])[1], ax=ax[1, i])
        matrix_plot(result_PR_thresholds, "Purples", r"$F_1$ score", metric_comp_func = lambda r: 2 * wrap(r[j])[0] * wrap(r[j])[1] / (wrap(r[j])[0] + wrap(r[j])[1]), ax=ax[2, i])
        ax[0, i].set_title(model)
        ax[1, i].set_title(model)
        ax[2, i].set_title(model)
    fig.tight_layout()
    fig.savefig(out_file_PR_OS)
    print("Saved to", out_file_PR_OS)

################
# UNUSED PLOTS #
################
'''fig, ax = plt.subplots(2, len(models), figsize=(sz * len(models), sz * 2))
for i, model in tqdm(enumerate(models)):
    output_path = os.path.join(path, model, "count_matched_quarks")
    if not os.path.exists(os.path.join(output_path, "result.pkl")):
        continue
    result = pickle.load(open(os.path.join(output_path, "result.pkl"), "rb"))
    #result_unmatched = pickle.load(open(os.path.join(output_path, "result_unmatched.pkl"), "rb"))
    result_fakes = pickle.load(open(os.path.join(output_path, "result_fakes.pkl"), "rb"))
    result_bc = pickle.load(open(os.path.join(output_path, "result_bc.pkl"), "rb"))
    result_PR = pickle.load(open(os.path.join(output_path, "result_PR.pkl"), "rb"))
    matrix_plot(result, "Blues", "Avg. matched dark quarks / event", ax=ax[0, i])
    matrix_plot(result_fakes, "Greens", "Avg. unmatched jets / event", ax=ax[1, i])
    ax[0, i].set_title(model)
    ax[1, i].set_title(model)
fig.tight_layout()
fig.savefig(out_file_avg_number_matched_quarks)
print("Saved to", out_file_avg_number_matched_quarks)'''

rinvs = [0.3, 0.5, 0.7]
sz = 4
fig, ax = plt.subplots(len(rinvs), 3, figsize=(3*sz, sz*len(rinvs)))
fig_AK, ax_AK = plt.subplots(len(rinvs), 3, figsize=(3*sz, sz*len(rinvs)))
fig_AK_ratio, ax_AK_ratio = plt.subplots(len(rinvs), 3, figsize=(3*sz, sz*len(rinvs)))


to_plot = {} # r_inv -> m_med -> precision, recall, R
to_plot_ak = {} # plotting for the AK baseline

### Plotting the score vs GT R plots

oranges = plt.get_cmap("Oranges")
reds = plt.get_cmap("Reds") # Plot a plot for each mass at given r_inv of the precision, recall, F1 score
purples = plt.get_cmap("Purples")

mDark = 20

for i, rinv in enumerate(rinvs):
    if rinv not in to_plot:
        to_plot[rinv] = {}
        to_plot_ak[rinv] = {}
    for j, model in enumerate(models):
        print("Model", model)
        if model not in radius:
            continue
        r = radius[model]
        output_path = os.path.join(path, model, "count_matched_quarks")
        if not os.path.exists(os.path.join(output_path, "result_PR.pkl")):
            continue
        result_PR = pickle.load(open(os.path.join(output_path, "result_PR.pkl"), "rb"))
        #if radius not in to_plot[rinv]:
        #    to_plot[rinv][radius] = {}
        for k, mMed in enumerate(sorted(result_PR.keys())):
            if mMed not in to_plot[rinv]:
                to_plot[rinv][mMed] = {"precision": [], "recall": [], "f1score": [], "R": []}
            precision = result_PR[mMed][mDark][rinv][0]
            recall = result_PR[mMed][mDark][rinv][1]
            f1score = 2 * precision * recall / (precision + recall)
            to_plot[rinv][mMed]["precision"].append(precision)
            to_plot[rinv][mMed]["recall"].append(recall)
            to_plot[rinv][mMed]["f1score"].append(f1score)
            to_plot[rinv][mMed]["R"].append(r)
    for mMed in sorted(to_plot[rinv].keys()):
        # normalize mmed between 0 and 1 (originally between 700 and 3000)
        mmed = (mMed - 500) / (3000 - 500)
        r = to_plot[rinv][mMed]
        print("Model R", r["R"])
        scatter_plot(ax[0, i], r["R"], r["precision"], label="m={} GeV".format(round(mMed)), color=oranges(mmed))
        scatter_plot(ax[1, i], r["R"], r["recall"], label="m={} GeV".format(round(mMed)), color=reds(mmed))
        scatter_plot(ax[2, i], r["R"], r["f1score"], label="m={} GeV".format(round(mMed)), color=purples(mmed))
    if not os.path.exists(os.path.join(ak_path, "result_PR_AKX.pkl")):
        continue
    result_PR_AKX = pickle.load(open(os.path.join(ak_path, "result_PR_AKX.pkl"), "rb"))
    result_jet_props_akx = pickle.load(open(os.path.join(ak_path, "result_jet_properties_AKX.pkl"), "rb"))
    #if radius not in to_plot[rinv]:
    #    to_plot[rinv][radius] = {}
    for k, mMed in enumerate(sorted(result_PR_AKX.keys())):
        if mMed not in to_plot_ak[rinv]:
            to_plot_ak[rinv][mMed] = {"precision": [], "recall": [], "f1score": [], "R": []}
        rs = sorted(result_PR_AKX[mMed][mDark][rinv].keys())
        precision = np.array([result_PR_AKX[mMed][mDark][rinv][k][0] for k in rs])
        recall = np.array([result_PR_AKX[mMed][mDark][rinv][k][1] for k in rs])
        f1score = 2 * precision * recall / (precision + recall)
        to_plot_ak[rinv][mMed]["precision"] += list(precision)
        to_plot_ak[rinv][mMed]["recall"] += list(recall)
        to_plot_ak[rinv][mMed]["f1score"] += list(f1score)
        to_plot_ak[rinv][mMed]["R"] += rs

    for mMed in sorted(to_plot_ak[rinv].keys()):
        # Normalize mmed between 0 and 1 (originally between 700 and 3000)
        mmed = (mMed - 500) / (3000 - 500)
        r = to_plot_ak[rinv][mMed]
        r_model = to_plot[rinv][mMed]
        print("AK R", r["R"])
        scatter_plot(ax_AK[0, i], r["R"], r["precision"], label="m={} GeV AK".format(round(mMed)), color=oranges(mmed), pattern=".--")
        scatter_plot(ax_AK[1, i], r["R"], r["recall"], label="m={} GeV AK".format(round(mMed)), color=reds(mmed), pattern=".--")
        scatter_plot(ax_AK[2, i], r["R"], r["f1score"], label="m={} GeV AK".format(round(mMed)), color=purples(mmed), pattern=".--")
        # r["R"] has more points than r_model["R"] - pick those from r["R"] that are in r_model["R"]
        r["R"] = np.array(r["R"])
        r["precision"] = np.array(r["precision"])
        r["recall"] = np.array(r["recall"])
        r["f1score"] = np.array(r["f1score"])
        filt = np.isin(r["R"], r_model["R"])
        r["R"] = r["R"][filt]
        r["precision"] = r["precision"][filt]
        r["recall"] = r["recall"][filt]
        r["f1score"] = r["f1score"][filt]
        scatter_plot(ax_AK_ratio[0, i], r["R"], r["precision"]/np.array(r_model["precision"]), label="m={} GeV AK".format(round(mMed)), color=oranges(mmed), pattern=".--")
        scatter_plot(ax_AK_ratio[1, i], r["R"], r["recall"]/np.array(r_model["recall"]), label="m={} GeV AK".format(round(mMed)), color=reds(mmed), pattern=".--")
        scatter_plot(ax_AK_ratio[2, i], r["R"], r["f1score"]/np.array(r_model["f1score"]), label="m={} GeV AK".format(round(mMed)), color=purples(mmed), pattern=".--")

    for ax1 in [ax, ax_AK, ax_AK_ratio]:
        ax1[0, i].set_title(f"Precision r_inv = {rinv}")
        ax1[1, i].set_title(f"Recall r_inv = {rinv}")
        ax1[2, i].set_title(f"F1 score r_inv = {rinv}")
        ax1[2, i].legend()
        ax1[1, i].legend()
        ax1[0, i].legend()
        ax1[0, i].grid()
        ax1[1, i].grid()
        ax1[2, i].grid()
        ax1[0, i].set_xlabel("GT R")
        ax1[1, i].set_xlabel("GT R")
        ax1[2, i].set_xlabel("GT R")
    ax_AK_ratio[0, i].set_ylabel("Precision (model=1)")
    ax_AK_ratio[1, i].set_ylabel("Recall (model=1)")
    ax_AK_ratio[2, i].set_ylabel("F1 score (model=1)")
fig.tight_layout()
fig_AK.tight_layout()
fig.savefig(os.path.join(get_path(args.input, "results"), "score_vs_GT_R_plots.pdf"))
fig_AK.savefig(os.path.join(get_path(args.input, "results"), "score_vs_GT_R_plots_AK.pdf"))
fig_AK_ratio.tight_layout()
fig_AK_ratio.savefig(os.path.join(get_path(args.input, "results"), "score_vs_GT_R_plots_AK_ratio.pdf"))

print("Saved to", os.path.join(get_path(args.input, "results"), "score_vs_GT_R_plots_AK.pdf"))
print("Saved to", os.path.join(get_path(args.input, "results"), "score_vs_GT_R_plots_AK_ratio.pdf"))

