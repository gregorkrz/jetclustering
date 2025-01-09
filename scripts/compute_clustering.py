import pickle
import torch
import os
import matplotlib.pyplot as plt
from src.utils.paths import get_path
from src.utils.utils import CPU_Unpickler
from pathlib import Path
from src.dataset.dataset import EventDataset
import numpy as np
import hdbscan
import argparse
from tqdm import tqdm

# filename = get_path("/work/gkrzmanc/jetclustering/results/train/Test_betaPt_BC_2025_01_03_15_07_14/eval_0.pkl", "results")
# for rinv=0.7, see /work/gkrzmanc/jetclustering/results/train/Test_betaPt_BC_rinv07_2025_01_03_15_38_58
# keeping the clustering script here for now, so that it's separated from the GPU-heavy tasks like inference (clustering may be changed frequently...)

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)

args = parser.parse_args()
path = get_path(args.input, "results")

#dir_results = get_path("/work/gkrzmanc/jetclustering/results/train/Test_betaPt_BC_2025_01_03_15_07_14/eval_0.pkl", "results")

def get_clustering_labels(coords, batch_idx, min_cluster_size=10, min_samples=20):
    labels = []
    for i in tqdm(np.unique(batch_idx)):
        filt = batch_idx == i
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                    cluster_selection_epsilon=0.1)
        cluster_labels = clusterer.fit_predict(coords[filt])
        labels.append(cluster_labels)
    return np.concatenate(labels)

for file in os.listdir(path):
    if file.startswith("eval_") and file.endswith(".pkl"):
        print("Computing clusters for file", file)
        result = CPU_Unpickler(open(os.path.join(path, file), "rb")).load()
        file_number = file.split("_")[1].split(".")[0]
        labels_path = os.path.join(path, "clustering_{}.pkl".format(file_number))
        if not os.path.exists(labels_path):
            #dataset = EventDataset.from_directory(result["filename"], mmap=True)
            if result["pred"].shape[1] == 4:
                labels = get_clustering_labels(result["pred"][:, :3], result["event_idx"])
            else:
                # L-GATr outputs 4-vectors
                labels = get_clustering_labels(result["pred"][:, :4], result["event_idx"])
            with open(labels_path, "wb") as f:
                pickle.dump(labels, f)
            print("Saved labels to", labels_path)
        else:
            print("Labels already exist for this file")
