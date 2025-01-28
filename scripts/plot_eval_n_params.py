import os
from tqdm import tqdm
import argparse
import pickle
from src.plotting.eval_matrix import matrix_plot, scatter_plot
from src.utils.paths import get_path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=False, default="scouting_PFNano_signals2/SVJ_hadronic_std/batch_eval/params_study")

args = parser.parse_args()
path = get_path(args.input, "results")

def get_short(network_config):
    if "transformer" in network_config.lower():
        return "Transformer"
    if "lgatr" in network_config.lower():
        return "LGATr"
    if "gatr" in network_config.lower():
        return "GATr"
    return "Unknown"

def get_model_details(path_to_eval):
    config = pickle.load(open(os.path.join(path_to_eval, "run_config.pkl"), "rb"))
    return config["num_parameters"], get_short(config["network_config"])

models = sorted([x for x in os.listdir(path) if not (os.path.isfile(os.path.join(path, x)) or "AK8" in x)])# + ["AK8", "AK8_GenJets"]
data = [get_model_details(os.path.join(path, model)) for model in models] + [(0, "AK8"), (0, "AK8_GenJets")]
idx = []


models = models + ["AK8", "AK8_GenJets"]

out_file_PR = os.path.join(get_path(args.input, "results"), "precision_recall_n_params.pdf")

sz = 5
fig, ax = plt.subplots(3, len(models), figsize=(sz * len(models), sz * 3))
result_scatter = {} # e.g. Transformer -> [xarr, yarr, yarr1, yarr2]
result_scatter_900_03 = {}
# n_params, P, R, f1

for i, model in tqdm(enumerate(models)):
    output_path = os.path.join(path, model, "count_matched_quarks")
    if not os.path.exists(os.path.join(output_path, "result.pkl")):
        continue
    result = pickle.load(open(os.path.join(output_path, "result.pkl"), "rb"))
    result_fakes = pickle.load(open(os.path.join(output_path, "result_fakes.pkl"), "rb"))
    result_bc = pickle.load(open(os.path.join(output_path, "result_bc.pkl"), "rb"))
    result_PR = pickle.load(open(os.path.join(output_path, "result_PR.pkl"), "rb"))
    #matrix_plot(result_PR, "Oranges", "Precision (N matched dark quarks / N predicted jets)", metric_comp_func = lambda r: r[0], ax=ax[0, i])
    #matrix_plot(result_PR, "Reds", "Recall (N matched dark quarks / N dark quarks)", metric_comp_func = lambda r: r[1], ax=ax[1, i])
    #matrix_plot(result_PR, "Purples", r"$F_1$ score", metric_comp_func = lambda r: 2 * r[0] * r[1] / (r[0] + r[1]), ax=ax[2, i])
    ax[0, i].set_title(str(data[i][0]) +  " " + data[i][1])
    ax[1, i].set_title(str(data[i][0]) +  " " + data[i][1])
    ax[2, i].set_title(str(data[i][0]) +  " " + data[i][1])
    if data[i][1] not in result_scatter:
        result_scatter[data[i][1]] = [[], [], [], []]
        result_scatter_900_03[data[i][1]] = [[], [], [], []]
    result_scatter[data[i][1]][0].append(data[i][0])
    pr = result_PR[700][20][0.7]
    pr_900_03 = result_PR[900][20][0.3]
    result_scatter[data[i][1]][3].append(2 * pr[0] * pr[1] / (pr[0] + pr[1]))
    result_scatter[data[i][1]][1].append(pr[0])
    result_scatter[data[i][1]][2].append(pr[1])
    result_scatter_900_03[data[i][1]][3].append(2 * pr_900_03[0] * pr_900_03[1] / (pr_900_03[0] + pr_900_03[1]))
    result_scatter_900_03[data[i][1]][1].append(pr_900_03[0])
    result_scatter_900_03[data[i][1]][2].append(pr_900_03[1])
    result_scatter_900_03[data[i][1]][0].append(data[i][0])

fig.tight_layout()
fig.savefig(out_file_PR)
print("Saved to", out_file_PR)

fig_scatter, ax_scatter = plt.subplots(3, 1, figsize=(sz , sz  * 3))

colors = {
    "Transformer": "green",
    "GATr": "blue",
    "LGATr": "red",
}
for key in result_scatter:
    scatter_plot(ax_scatter[0], result_scatter[key][0], result_scatter[key][1], key)
    scatter_plot(ax_scatter[1], result_scatter[key][0], result_scatter[key][2], key)
    scatter_plot(ax_scatter[2], result_scatter[key][0], result_scatter[key][3], key)

ax_scatter[0].set_ylabel("Precision")
ax_scatter[1].set_ylabel("Recall")
ax_scatter[2].set_ylabel("F1 score")
ax_scatter[0].set_xlabel("N params")
ax_scatter[1].set_xlabel("N params")
ax_scatter[2].set_xlabel("N params")
ax_scatter[0].legend()
ax_scatter[1].legend()
ax_scatter[2].legend()
ax_scatter[0].grid()
ax_scatter[1].grid()
ax_scatter[2].grid()
ax_scatter[0].set_xscale("log")
ax_scatter[1].set_xscale("log")
ax_scatter[2].set_xscale("log")

fig_scatter.tight_layout()
fig_scatter.savefig(out_file_PR.replace(".pdf", "_scatter_700_07.pdf"))
print("Saved to", out_file_PR.replace(".pdf", "_scatter_700_07.pdf"))

fig_scatter, ax_scatter = plt.subplots(3, 1, figsize=(sz, sz*3))

for key in result_scatter_900_03:
    scatter_plot(ax_scatter[0], result_scatter_900_03[key][0], result_scatter_900_03[key][1], key)
    scatter_plot(ax_scatter[1], result_scatter_900_03[key][0], result_scatter_900_03[key][2], key)
    scatter_plot(ax_scatter[2], result_scatter_900_03[key][0], result_scatter_900_03[key][3], key)

ax_scatter[0].set_ylabel("Precision")
ax_scatter[1].set_ylabel("Recall")
ax_scatter[2].set_ylabel("F1 score")
ax_scatter[0].set_xlabel("N params")
ax_scatter[1].set_xlabel("N params")
ax_scatter[2].set_xlabel("N params")
ax_scatter[0].legend()
ax_scatter[1].legend()
ax_scatter[2].legend()
ax_scatter[0].grid()
ax_scatter[1].grid()
ax_scatter[2].grid()
ax_scatter[0].set_xscale("log")
ax_scatter[1].set_xscale("log")
ax_scatter[2].set_xscale("log")

fig_scatter.tight_layout()
fig_scatter.savefig(out_file_PR.replace(".pdf", "_scatter_900_03.pdf"))
print("Saved to", out_file_PR.replace(".pdf", "_scatter_900_03.pdf"))


