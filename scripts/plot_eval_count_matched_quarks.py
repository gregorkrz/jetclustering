import os
from tqdm import tqdm
import argparse
import pickle
from src.plotting.eval_matrix import matrix_plot, scatter_plot
from src.utils.paths import get_path
import matplotlib.pyplot as plt
import numpy as np


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

fig, ax = plt.subplots(3, nplots, figsize=(nplots*sz, 3*sz))
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

print("Saved to", os.path.join(get_path(args.input, "results"), "precision_recall_thresholds.pdf"))

sz = 5

fig, ax = plt.subplots(3, len(models), figsize=(sz * len(models), sz * 3))
for i, model in tqdm(enumerate(models)):
    output_path = os.path.join(path, model, "count_matched_quarks")
    ak_path = os.path.join(path, "AKX", "count_matched_quarks")

    if not os.path.exists(os.path.join(output_path, "result.pkl")):
        continue
    result = pickle.load(open(os.path.join(output_path, "result.pkl"), "rb"))
    #result_unmatched = pickle.load(open(os.path.join(output_path, "result_unmatched.pkl"), "rb"))
    result_fakes = pickle.load(open(os.path.join(output_path, "result_fakes.pkl"), "rb"))
    result_bc = pickle.load(open(os.path.join(output_path, "result_bc.pkl"), "rb"))
    result_PR = pickle.load(open(os.path.join(output_path, "result_PR.pkl"), "rb"))
    #matrix_plot(result, "Blues", "Avg. matched dark quarks / event").savefig(os.path.join(output_path, "avg_matched_dark_quarks.pdf"), ax=ax[0, i])
    #matrix_plot(result_fakes, "Greens", "Avg. unmatched jets / event").savefig(os.path.join(output_path, "avg_unmatched_jets.pdf"), ax=ax[1, i])
    matrix_plot(result_PR, "Oranges", "Precision (N matched dark quarks / N predicted jets)", metric_comp_func = lambda r: r[0], ax=ax[0, i])
    matrix_plot(result_PR, "Reds", "Recall (N matched dark quarks / N dark quarks)", metric_comp_func = lambda r: r[1], ax=ax[1, i])
    matrix_plot(result_PR, "Purples", r"$F_1$ score", metric_comp_func = lambda r: 2 * r[0] * r[1] / (r[0] + r[1]), ax=ax[2, i])
    ax[0, i].set_title(model)
    ax[1, i].set_title(model)
    ax[2, i].set_title(model)
fig.tight_layout()
fig.savefig(out_file_PR)
print("Saved to", out_file_PR)

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
        print("AK R", r["R"])
        r = to_plot_ak[rinv][mMed]
        r_model = to_plot[rinv][mMed]
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
