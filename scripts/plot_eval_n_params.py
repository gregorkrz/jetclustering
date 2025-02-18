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

def get_steps(config):
    if "ckpt_step" in config:
        return config["ckpt_step"]
    # else, config["load_model_weights"] looks like /.../.../step_xxxx_epoch_y.ckpt (fallback)
    return int(config["load_model_weights"].split("/")[-1].split("_")[1])

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
    return config["num_parameters"], get_short(config["network_config"]), get_steps(config)

models = sorted([x for x in os.listdir(path) if not (os.path.isfile(os.path.join(path, x)) or "AK8" in x)])# + ["AK8", "AK8_GenJets"]
data = [get_model_details(os.path.join(path, model)) for model in models] + [(0, "AK8", 0), (0, "AK8_GenJets", 0)]
models = models + ["AK8", "AK8_GenJets"]

out_file_PR = os.path.join(get_path(args.input, "results"), "precision_recall_n_params.pdf")

sz = 5
fig, ax = plt.subplots(3, len(models), figsize=(sz * len(models), sz * 3))
result_scatter = {} # e.g. Transformer -> [xarr, yarr, yarr1, yarr2]
result_scatter_900_03 = {}

result_by_step = {"900_03": {}, "700_07": {}} # Model+n_params -> [step, p, r, f1]

def get_arch_name(n_params, net_short):
    if net_short == "Transformer":
        if n_params == 4674:
            return "Tr-2-16-4"
        elif n_params == 1201108:
            return "Tr"
        elif n_params == 1322274:
            return "Tr"
        elif n_params == 167394:
            return "Tr-5-64-4"
    if net_short == "LGATr":
        if n_params == 8424:
            return "LGATr-2-4-4"
        elif n_params == 1201108:
            return "LGATr"
        elif n_params == 156332:
            return "LGATr-3-16-16"
    if net_short == "GATr":
        if n_params == 6533:
            return "GATr-2-4-4"
        if n_params == 926041:
            return "GATr"
    if "AK8" in net_short:
        return net_short
    return None
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
    arch = get_arch_name(data[i][0], data[i][1])
    if arch is not None:
        if result_by_step["900_03"].get(arch) is None:
            for key in result_by_step:
                result_by_step[key][arch] = [[], [], [], []]
        pr = result_PR[900][20][0.3]
        result_by_step["900_03"][arch][0].append(data[i][2])
        result_by_step["900_03"][arch][1].append(pr[0])
        result_by_step["900_03"][arch][2].append(pr[1])
        result_by_step["900_03"][arch][3].append(2 * pr[0] * pr[1] / (pr[0] + pr[1]))
        pr = result_PR[700][20][0.7]
        result_by_step["700_07"][arch][0].append(data[i][2])
        result_by_step["700_07"][arch][1].append(pr[0])
        result_by_step["700_07"][arch][2].append(pr[1])
        result_by_step["700_07"][arch][3].append(2 * pr[0] * pr[1] / (pr[0] + pr[1]))

    if data[i][2] != 40000:
        continue
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


fig_scatter, ax_scatter = plt.subplots(3, 2, figsize=(sz*2, sz*3))

for i, key in enumerate(sorted(list(result_by_step.keys()))):
    for model in result_by_step[key]:
        #scatter_plot(ax_scatter[], result_scatter_900_03[key][0], result_scatter_900_03[key][1], key)
        #scatter_plot(ax_scatter[1], result_scatter_900_03[key][0], result_scatter_900_03[key][2], key)
        #scatter_plot(ax_scatter[2], result_scatter_900_03[key][0], result_scatter_900_03[key][3], key)
        if "AK8" in model:
            # put a horizontal dotted line instead of a scatterplot, as there is only one dot
            colors = {"AK8": "gray", "AK8_GenJets": "black"}
            ax_scatter[0, i].axhline(result_by_step[key][model][1][0], label=model, color=colors[model], linestyle="--")
            ax_scatter[1, i].axhline(result_by_step[key][model][2][0], label=model, color=colors[model], linestyle="--")
            ax_scatter[2, i].axhline(result_by_step[key][model][3][0], label=model, color=colors[model], linestyle="--")
        else:
            scatter_plot(ax_scatter[0, i], result_by_step[key][model][0], result_by_step[key][model][1], model)
            scatter_plot(ax_scatter[1, i], result_by_step[key][model][0], result_by_step[key][model][2], model)
            scatter_plot(ax_scatter[2, i], result_by_step[key][model][0], result_by_step[key][model][3], model)
        ax_scatter[0, i].set_title(key)
        ax_scatter[1, i].set_title(key)
        ax_scatter[2, i].set_title(key)
        ax_scatter[0, i].set_ylabel("Precision")
        ax_scatter[1, i].set_ylabel("Recall")
        ax_scatter[2, i].set_ylabel("F_1 score")
        ax_scatter[0, i].set_xlabel("training steps")
        ax_scatter[1, i].set_xlabel("training steps")
        ax_scatter[2, i].set_xlabel("training steps")
        ax_scatter[0, i].legend()
        ax_scatter[1, i].legend()
        ax_scatter[2, i].legend()
        ax_scatter[0, i].grid()
        ax_scatter[1, i].grid()
        ax_scatter[2, i].grid()
        ax_scatter[0, i].set_xscale("log")
        ax_scatter[1, i].set_xscale("log")
        ax_scatter[2, i].set_xscale("log")
fig_scatter.tight_layout()
fig_scatter.savefig(out_file_PR.replace(".pdf", "_by_step.pdf"))
print("Saved to", out_file_PR.replace(".pdf", "_by_step.pdf"))

