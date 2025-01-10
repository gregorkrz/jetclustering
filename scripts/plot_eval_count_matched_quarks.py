import os
from tqdm import tqdm
import argparse
import pickle
from src.plotting.eval_matrix import matrix_plot
from src.utils.paths import get_path
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=False, default="scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval")

args = parser.parse_args()
path = get_path(args.input, "results")

models = sorted([x for x in os.listdir(path) if not os.path.isfile(os.path.join(path, x))])

out_file_PR = os.path.join(get_path(args.input, "results"), "precision_recall.pdf")
out_file_avg_number_matched_quarks = os.path.join(get_path(args.input, "results"), "avg_number_matched_quarks.pdf")
sz = 5
fig, ax = plt.subplots(3, len(models), figsize=(sz * len(models), sz * 3))

for i, model in tqdm(enumerate(models)):
    output_path = os.path.join(path, model, "count_matched_quarks")
    result = pickle.load(open(os.path.join(output_path, "result.pkl"), "rb"))
    result_unmatched = pickle.load(open(os.path.join(output_path, "result_unmatched.pkl"), "rb"))
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
    result = pickle.load(open(os.path.join(output_path, "result.pkl"), "rb"))
    result_unmatched = pickle.load(open(os.path.join(output_path, "result_unmatched.pkl"), "rb"))
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
