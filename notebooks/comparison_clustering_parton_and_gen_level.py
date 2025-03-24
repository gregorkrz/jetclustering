import pickle
import torch
import os
import matplotlib.pyplot as plt
from src.utils.paths import get_path
from src.utils.utils import CPU_Unpickler
from pathlib import Path
import fastjet
from src.dataset.dataset import EventDataset
import numpy as np
from src.plotting.plot_coordinates import plot_coordinates


filename_parton_level = get_path("/work/gkrzmanc/jetclustering/results/train/Eval_no_pid_eval_1_2025_03_05_14_41_16/eval_1.pkl", "results")
result_parton_level = CPU_Unpickler(open(filename_parton_level, "rb")).load()
dataset_parton_level = EventDataset.from_directory(result_parton_level["filename"], mmap=True)

filename_gen_level = get_path("/work/gkrzmanc/jetclustering/results/train/Eval_no_pid_eval_1_2025_03_05_14_40_30/eval_1.pkl", "results")
result_gen_level = CPU_Unpickler(open(filename_gen_level, "rb")).load()
dataset_gen_level = EventDataset.from_directory(result_gen_level["filename"], mmap=True)

filename_pfcands_level = get_path("/work/gkrzmanc/jetclustering/results/train/Eval_no_pid_eval_1_2025_03_05_14_41_38/eval_1.pkl", "results")
result_pfcands_level = CPU_Unpickler(open(filename_pfcands_level, "rb")).load()
dataset_pfcands_level = EventDataset.from_directory(result_pfcands_level["filename"], mmap=True)

EVENT_ID=15

# plotly 3d plot of result["pred"], colored by result["GT_cluster"]
def plot_result(result, dataset_path, save_dir):
    filt = result["event_idx"] == EVENT_ID
    # normalized coordinates
    norm_coords = result["pred"][filt, 1:4] #/ np.linalg.norm(result["pred"][filt, 1:4] , axis=1 ,keepdims=1)
    pt = torch.tensor(result["pt"][filt])
    clusters_file = get_path(os.path.join(dataset_path, f"clustering_hdbscan_4_05_1.pkl"), "results")
        #clusters_file=None
    model_clusters = CPU_Unpickler(open(clusters_file, "rb")).load()# torch.tensor(model_clusters[filt])
    plot_coordinates(norm_coords, pt, result["GT_cluster"][filt]).write_html(save_dir)
    print("-----")


plot_result(result_parton_level, "/work/gkrzmanc/jetclustering/results/train/Eval_no_pid_eval_1_2025_03_05_14_41_16", "/work/gkrzmanc/jetclustering/results/GT_color_parton_level_{}.html".format(EVENT_ID))
plot_result(result_gen_level, "/work/gkrzmanc/jetclustering/results/train/Eval_no_pid_eval_1_2025_03_05_14_40_30", "/work/gkrzmanc/jetclustering/results/GT_color_gen_level_{}.html".format(EVENT_ID))
plot_result(result_pfcands_level, "/work/gkrzmanc/jetclustering/results/train/Eval_no_pid_eval_1_2025_03_05_14_41_38", "/work/gkrzmanc/jetclustering/results/GT_color_pfcands_level_{}.html".format(EVENT_ID))


