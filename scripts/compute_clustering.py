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

filename = get_path("/work/gkrzmanc/jetclustering/results/train/Test_betaPt_BC_2025_01_03_15_07_14/eval_0.pkl", "results")
# for rinv=0.7, see /work/gkrzmanc/jetclustering/results/train/Test_betaPt_BC_rinv07_2025_01_03_15_38_58

result = CPU_Unpickler(open(filename, "rb")).load()
dataset = EventDataset.from_directory(result["filename"], mmap=True)

from tqdm import tqdm
def get_clustering_labels(coords, batch_idx, min_cluster_size=10, min_samples=20):
    labels = []
    for i in tqdm(np.unique(batch_idx)):
        filt = batch_idx == i
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                    cluster_selection_epsilon=0.1)
        cluster_labels = clusterer.fit_predict(coords[filt])
        labels.append(cluster_labels)
    return np.concatenate(labels)

labels = get_clustering_labels(result["pred"][:, :3], result["event_idx"])
labels_path = os.path.join(os.path.dirname(filename), "HDBSCAN_10_20.pkl")

with open(labels_path, "wb") as f:
    pickle.dump(labels, f)

