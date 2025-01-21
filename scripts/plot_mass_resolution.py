import os
from tqdm import tqdm
import argparse
import pickle
from src.plotting.eval_matrix import matrix_plot, scatter_plot
from src.utils.paths import get_path
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=False, default="scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R")

args = parser.parse_args()
path = get_path(args.input, "results")

models = sorted([x for x in os.listdir(path) if not os.path.isfile(os.path.join(path, x))])

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
}

out_file = {}

sz = 5
fig, ax = plt.subplots(len(models), 2, figsize=(sz * 2, sz/2 * len(models)))

bins = np.linspace(0, 2, 100)
for i, model in tqdm(enumerate(models)):
    output_path = os.path.join(path, model, "count_matched_quarks")
    f = os.path.join(output_path, "result_m.pkl")
    if not os.path.isfile(f):
        continue
    result = pickle.load(open(f, "rb"))
    f1 = os.path.join(output_path, "result_PR.pkl")
    result1 = pickle.load(open(f1, "rb"))
    r = result[900][20][0.3]
    r1 = result1[900][20][0.3][1] # n jets
    ax[i, 0].hist(r["m_pred"] / r["m_true"], bins=bins, histtype="step")
    ax[i, 1].hist(r["mt_pred"] / r["mt_true"], bins=bins, histtype="step")
    ax[i, 0].set_title(model)
    ax[i, 1].set_title(model)
    ax[i, 0].set_xlabel("m_pred / m_true")
    ax[i, 1].set_xlabel("mt_pred / mt_true")
    ax[i, 0].set_yscale("log")
    ax[i, 1].set_yscale("log")
fig.tight_layout()
fig.savefig(os.path.join(path, "mass_histograms.pdf"))



#######

sz = 5
r_invs = {"03": 0.3, "07": 0.7, "05": 0.5}
c = {}
for r_inv in r_invs:
    fig, ax = plt.subplots(len(result), 2, figsize=(sz * 2, sz/2 * len(models)))
    bins = np.linspace(0, 2, 100)
    for i, mmed in tqdm(enumerate(sorted(result.keys()))):
        for j, model in enumerate(models):
            output_path = os.path.join(path, model, "count_matched_quarks")
            f = os.path.join(output_path, "result_m.pkl")
            if not os.path.isfile(f):
                continue
            if f not in c:
                c[f] = pickle.load(open(f, "rb"))
            result = c[f]
            r = result[mmed][20][r_invs[r_inv]]
            ax[i, 0].hist(r["m_pred"] / r["m_true"], bins=bins, histtype="step", label=model)
            ax[i, 1].hist(r["mt_pred"] / r["mt_true"], bins=bins, histtype="step", label=model)
        ax[i, 0].set_title("m_med = " + str(mmed))
        ax[i, 1].set_title("m_med = " + str(mmed))
        ax[i, 0].set_xlabel("m_pred / m_true")
        ax[i, 1].set_xlabel("mt_pred / mt_true")
        ax[i, 0].set_yscale("log")
        ax[i, 1].set_yscale("log")
        ax[i, 0].legend()
        ax[i, 1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(path, "mass_histograms_model_comparison_{}.pdf".format(r_inv)))

##########

