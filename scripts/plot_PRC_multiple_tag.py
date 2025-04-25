import os
from tqdm import tqdm
import argparse
import pickle
from src.plotting.eval_matrix import matrix_plot, scatter_plot
from src.utils.paths import get_path
import matplotlib.pyplot as plt
import numpy as np

inputs = {
    "Delphes": "Delphes_020425_test/batch_eval_2k/DelphesTestCMSTrain",
    "CMS FullSim": "Feb26_2025_E1000_N500_noPartonFilter_GluonFix_Small2K_F_part0/batch_eval_2k/SmallDSReprod2"
}


files = {
    key: pickle.load(open(os.path.join(get_path(value, "results"), "precision_recall.pkl"), "rb")) for key, value in inputs.items()
}
titles = {key: set(value.keys()) for key, value in files.items()}
# make a set of the intersections of titles
intersections = sorted(list(set.intersection(*titles.values())))


output_dirs = []
for _, value in inputs.items():
    output_dirs.append(os.path.join(get_path(value, "results"), "comparison.pdf"))

min_file_len = 999
for key, value in files.items():
    if len(value) < min_file_len:
        min_file_len = len(value)

sz = 5.5

fig, ax = plt.subplots(len(inputs), min_file_len, figsize=(sz*min_file_len, sz*len(inputs)))

for i, key in enumerate(sorted(files.keys())):
    for j, title in enumerate(intersections):
        matrix = files[key][title]
        matrix_plot(matrix, "Purples", r"$F_1$ score", metric_comp_func=lambda r: 2 * r[0] * r[1] / (r[0] + r[1]), ax=ax[i, j])
        ax[i, j].set_title(key + " " + title)

for f in output_dirs:
    fig.tight_layout()
    fig.savefig(f)
    print("saved to", f)


