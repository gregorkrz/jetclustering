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
fig, ax = plt.subplots(len(models), 2, figsize=(sz * 2, sz * len(models)))

bins = np.linspace(0, 2, 200)
for i, model in tqdm(enumerate(models)):
    output_path = os.path.join(path, model, "count_matched_quarks")
    f = os.path.join(output_path, "result_m.pkl")
    if not os.path.isfile(f):
        continue
    result = pickle.load(open(f, "rb"))
    r = result[900][20][0.3]
    ax[i, 0].hist(r["m_pred"] / r["m_true"], bins=bins)
    ax[i, 1].hist(r["mt_pred"] / r["mt_true"], bins=bins)
    ax[i, 0].set_title(model)
    ax[i, 1].set_title(model)
    ax[i, 0].set_xlabel("m_pred / m_true")
    ax[i, 1].set_xlabel("mt_pred / mt_true")
fig.tight_layout()
fig.savefig(os.path.join(path, "mass_histograms.pdf"))
