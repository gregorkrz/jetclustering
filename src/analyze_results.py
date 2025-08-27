import pickle
import torch
import os
import matplotlib.pyplot as plt
from src.utils.paths import get_path
from src.utils.utils import CPU_Unpickler
from pathlib import Path
from src.plotting.histograms import score_histogram, per_pt_score_histogram, plot_roc_curve, confusion_matrix_plot
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
args = parser.parse_args()

input_dir = get_path(args.input, "results")


def plot_score_histograms(result, eval_path):
    pt = result["pt"]
    y_true = (result["GT_cluster"] >= 0)
    y_pred = result["pred"][:, -1]
    score_histogram(y_true, y_pred, sz=5).savefig(os.path.join(eval_path, "binary_classifier_scores.pdf"))
    per_pt_score_histogram(y_true, y_pred, pt).savefig(os.path.join(eval_path, "binary_classifier_scores_per_pt.pdf"))
    plot_roc_curve(y_true, y_pred).savefig(os.path.join(eval_path, "roc_curve.pdf"))

import numpy as np

def plot_four_momentum_spectrum(result, eval_path):
    y_true = (result["GT_cluster"] >= 0)
    y_pred = result["pred"][:, :4]
    mass_squared = y_pred[:, 0]**2 - y_pred[:, 1]**2 - y_pred[:, 2]**2 - y_pred[:, 3]**2
    signal_masses = mass_squared[y_true]
    bkg_masses = mass_squared[~y_true]
    all_masses = mass_squared
    fig, ax = plt.subplots()
    bins = np.linspace(-25, 25, 200)
    #ax.hist(signal_masses, bins=bins, histtype="step", label="Signal")
    #ax.hist(bkg_masses, bins=bins, histtype="step", label="Background")
    ax.hist(all_masses, bins=bins, histtype="step", label="All")
    ax.set_xlabel("m^2")
    ax.set_yscale("log")
    ax.set_ylabel("count")
    ax.legend()
    fig.savefig(os.path.join(eval_path, "mass_squared.pdf"))

def plot_cm(result, eval_path):
    # Confusion matrices
    y_true = (result["GT_cluster"] >= 0)
    y_pred = result["pred"][:, 3]
    pt = result["pt"]
    sz = 5
    fig, ax = plt.subplots(1, 3, figsize=(3*sz/2, sz/2))
    confusion_matrix_plot(y_true, y_pred > 0.5, ax[0])
    ax[0].set_title("Classifier (cut at 0.5)")
    confusion_matrix_plot(y_true, result["radius_cluster_FatJets"], ax[2])
    ax[2].set_title("FatJets")
    confusion_matrix_plot(y_true, result["radius_cluster_GenJets"], ax[1])
    ax[1].set_title("GenJets")
    fig.tight_layout()
    fig.savefig(os.path.join(eval_path, "confusion_matrix.pdf"))

for file in os.listdir(input_dir):
    print("File:", file)
    filename = get_path(os.path.join(input_dir, file),"results")
    if file.startswith("eval_") and file.endswith(".pkl"):
        print("Plotting file", filename)
        result = CPU_Unpickler(open(filename, "rb")).load()
        eval_path = os.path.join(os.path.dirname(filename), "full_eval_" + file.split("_")[1].split(".")[0])

        print(result.keys())
        Path(eval_path).mkdir(parents=True, exist_ok=True)

        def plotting_blueprint(result, eval_path):
            pass

        #plotting_jobs = [plot_score_histograms, plot_cm]
        plotting_jobs = [plot_four_momentum_spectrum]
        from time import time

        for job in plotting_jobs:
            t0 = time()
            print("Starting plotting job", job.__name__)
            try:
                job(result, eval_path)
            except Exception as e:
                print(f"Error in {job.__name__}: {e}")
                # print the traceback of the exception
                import traceback
                traceback.print_exc()

            print(f"{job.__name__} took {time()-t0:.2f}s")

