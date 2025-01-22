import os
from tqdm import tqdm
import argparse
import pickle
from src.plotting.eval_matrix import matrix_plot, scatter_plot
from src.utils.paths import get_path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=False, default="scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval")

args = parser.parse_args()
path = get_path(args.input, "results")

models = sorted([x for x in os.listdir(path) if not os.path.isfile(os.path.join(path, x))])
models = [x for x in models if "C1010" not in x]

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



out_file_PR = os.path.join(get_path(args.input, "results"), "precision_recall.pdf")
out_file_avg_number_matched_quarks = os.path.join(get_path(args.input, "results"), "avg_number_matched_quarks.pdf")
sz = 5
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

fig, ax = plt.subplots(2, len(models), figsize=(sz * len(models), sz * 2))
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
print("Saved to", out_file_avg_number_matched_quarks)

rinvs = [0.3, 0.5, 0.7]
sz = 4
fig, ax = plt.subplots(len(rinvs), 3, figsize=(3*sz, sz*len(rinvs)))

to_plot = {} # r_inv -> m_med -> precision, recall, R

# Plot a plot for each mass at given rinv of the precision, recall, F1 score
oranges = plt.get_cmap("Oranges")
reds = plt.get_cmap("Reds")
purples = plt.get_cmap("Purples")

mDark = 20

for i, rinv in enumerate(rinvs):
    if rinv not in to_plot:
        to_plot[rinv] = {}
    for j, model in enumerate(models):
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
        scatter_plot(ax[0, i], r["R"], r["precision"], label="m={} GeV".format(round(mMed)), color=oranges(mmed))
        scatter_plot(ax[1, i], r["R"], r["recall"], label="m={} GeV".format(round(mMed)), color=reds(mmed))
        scatter_plot(ax[2, i], r["R"], r["f1score"], label="m={} GeV".format(round(mMed)), color=purples(mmed))
    ax[0, i].set_title(f"Precision r_inv = {rinv}")
    ax[1, i].set_title(f"Recall r_inv = {rinv}")
    ax[2, i].set_title(f"F1 score r_inv = {rinv}")
    ax[2, i].legend()
    ax[1, i].legend()
    ax[0, i].legend()
    ax[0, i].grid()
    ax[1, i].grid()
    ax[2, i].grid()
    ax[0, i].set_xlabel("GT R")
    ax[1, i].set_xlabel("GT R")
    ax[2, i].set_xlabel("GT R")
fig.tight_layout()
fig.savefig(os.path.join(get_path(args.input, "results"), "score_vs_GT_R_plots.pdf"))

