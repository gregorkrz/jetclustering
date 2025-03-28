import pickle
import torch
import os
import matplotlib.pyplot as plt
from src.utils.paths import get_path
from src.utils.utils import CPU_Unpickler
from pathlib import Path
from src.dataset.dataset import EventDataset
import numpy as np
from src.plotting.plot_event import plot_event
from pathlib import Path

#%%

def get_properties(name):
    # get mediator mass, dark quark mass, r_inv from the filename
    parts = name.strip().strip("/").split("/")[-1].split("_")
    try:
        mMed = int(parts[1].split("-")[1])
        mDark = int(parts[2].split("-")[1])
        rinv = float(parts[3].split("-")[1])
    except:
        # another convention
        mMed = int(parts[2].split("-")[1])
        mDark = int(parts[3].split("-")[1])
        rinv = float(parts[4].split("-")[1])
    return mMed, mDark, rinv


#%%

clist = ['#1f78b4', '#b3df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbe6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
colors = {
    -1: "gray",
    0: clist[0],
    1: clist[1],
    2: clist[2],
    3: clist[3],
    4: clist[4],
    5: clist[5],
    6: clist[6],
    7: clist[7],
}

#%%
# The 'default' models:

models = {
    "GATr_rinv_03_m_900": "train/Test_betaPt_BC_all_datasets_2025_01_07_17_50_45",
    "GATr_rinv_07_m_900": "train/Test_betaPt_BC_all_datasets_2025_01_08_10_54_58",
    #"LGATr_rinv_03_m_900": "train/Test_LGATr_all_datasets_2025_01_08_19_27_54",
    "LGATr_rinv_07_m_900_s31k": "train/Eval_LGATr_SB_spatial_part_only_1_2025_01_13_14_31_58"
}

# Models with the varying R study

models = {
    #"R06": "train/Eval_GT_R_lgatr_R06_2025_01_16_13_41_48",
    #"R07": "train/Eval_GT_R_lgatr_R07_2025_01_16_13_41_41",
    #"R09": "train/Eval_GT_R_lgatr_R09_2025_01_16_13_41_45",
    "R=0.8": "train/Test_LGATr_all_datasets_2025_01_08_19_27_54",
    "R=1.0": "train/Eval_GT_R_lgatr_R10_2025_01_16_13_41_52",
    "R=1.4": "train/Eval_GT_R_lgatr_R14_2025_01_18_13_28_47",
    "R=2.0": "train/Eval_GT_R_lgatr_R20_2025_01_22_10_51_30"
}


## Objectness score odels

models = {
    "R=2.0,OS_GT=closest_only": "train/Eval_objectness_score_2025_02_14_11_10_14",
    "R=2.0,GT=all_in_radius": "train/Eval_objectness_score_2025_02_12_15_34_33",
    "R=0.8,GT=all_in_radius": "train/Eval_objectness_score_2025_02_10_14_59_49"
}

# Parton-level, gen-level and scouting PFCands models
models = {
    "parton-level": "train/Eval_no_pid_eval_2025_03_04_15_55_38",
    "gen-level": "train/Eval_no_pid_eval_2025_03_04_15_54_50",
    "scouting": "train/Eval_no_pid_eval_2025_03_04_16_06_57"
}

# Parton-level, gen-level and scouting PFCands models
models = {
    "parton-level": "train/Eval_no_pid_eval_1_2025_03_05_14_41_16",
    "gen-level": "train/Eval_no_pid_eval_1_2025_03_05_14_40_30",
    "scouting": "train/Eval_no_pid_eval_1_2025_03_05_14_41_38"
}

models = {
    "parton-level": "train/Eval_no_pid_eval_full_1_2025_03_18_16_56_02",
    "scouting": "train/Eval_no_pid_eval_full_1_2025_03_17_21_19_22",
    "gen-level": "train/Eval_no_pid_eval_full_1_2025_03_18_16_45_41"
}

# Trained on all data!

models1 = {
    "parton-level": "train/Eval_no_pid_eval_full_1_2025_03_17_23_44_49",
    "scouting PFCands": "train/Eval_no_pid_eval_full_1_2025_03_18_15_31_41",
    "gen-level": "train/Eval_no_pid_eval_full_1_2025_03_18_15_31_58"
}


# Trained on 900_03, but evaluated with eta and pt filters for the particles
models = {
    "parton-level": "train/Eval_eval_19March2025_2025_03_19_22_08_15",
    "scouting PFCands": "train/Eval_eval_19March2025_2025_03_19_22_08_22",
    "gen-level": "train/Eval_eval_19March2025_2025_03_19_22_08_18"
}

import wandb
api = wandb.Api()

def get_eval_run_names(tag):
    # from the api, get all the runs with the tag that are finished
    runs = api.runs(
        path="fcc_ml/svj_clustering",
        filters={"tags": {"$in": [tag.strip()]}}
    )
    return [run.name for run in runs if run.state == "finished"], [run.config for run in runs if run.state == "finished"]

def get_run_by_name(name):
    runs = api.runs(
        path="fcc_ml/svj_clustering",
        filters={"display_name": {"$eq": name.strip()}}
    )
    runs = api.runs(
        path="fcc_ml/svj_clustering",
        filters={"display_name": {"$eq": name.strip()}}
    )
    if runs.length != 1:
        return None
    return runs[0]

def get_models_from_tag(tag):
    models = {}
    for run in get_eval_run_names(tag)[0]:
        print("Run:", run)
        run = get_run_by_name(run)
        if run.config["parton_level"]:
            name = "parton-level"
        elif run.config[("gen_level")]:
            name = "gen-level"
        else:
            name = "scouting PFCands"
            if run.config["augment_soft_particles"]:
                name += " (soft part.)"
            if run.config["gt_radius"]:
                name += " GT_R=" + str(run.config["gt_radius"])
            if "transformer" in run.config["network_config"]:
                name += " (T)"
        models[name] = "train/" + run.name
    return models


models = get_models_from_tag("eval_19March2025_small_aug")
print(models)
# R = 2.0 models
#models = {
#    "parton-level": "train/Eval_eval_19March2025_2025_03_19_22_55_48",
#    "gen-level": "train/Eval_eval_19March2025_2025_03_19_23_20_01",
#    "scouting PFCands": "train/Eval_eval_19March2025_2025_03_19_23_43_07"
#}

output_path = get_path("eval_19March2025_small_aug", "results")

Path(output_path).mkdir(parents=1, exist_ok=1)

sz = 3
n_events_per_file = 10
# len(models) columns, n_events_per_file rows
from src.layers.object_cond import calc_eta_phi

for ds in range(5):
    print("-------- DS:", ds)
    fig, ax = plt.subplots(n_events_per_file, len(models) * 2,
                           figsize=(len(models) * sz * 2, n_events_per_file * sz))
    # also one only with real coordinates
    fig1, ax1 = plt.subplots(n_events_per_file, len(models),
                            figsize=(len(models) * sz, n_events_per_file * sz))
    for mn, model in enumerate(sorted(models.keys())):
        print("    -------- model:", model)
        dataset_path = models[model]
        filename = get_path(os.path.join(dataset_path, f"eval_{str(ds)}.pkl"), "results", fallback=1)
        clusters_file = get_path(os.path.join(dataset_path, f"clustering_hdbscan_4_05_{str(ds)}.pkl"), "results", fallback=1)
        #clusters_file=None
        if not os.path.exists(filename):
            print("File does not exist:", filename)
            continue
        result = CPU_Unpickler(open(filename, "rb")).load()
        print(result["filename"])
        m_med, m_dark, r_inv = get_properties(result["filename"])
        if os.path.exists(clusters_file):
            clusters = CPU_Unpickler(open(clusters_file, "rb")).load()
        else:
            clusters = result["model_cluster"].numpy()
            clusters_file = None
        dataset = EventDataset.from_directory(result["filename"], mmap=True, model_output_file=filename, model_clusters_file=clusters_file, include_model_jets_unfiltered=True, aug_soft="soft part" in model)
        for e in range(n_events_per_file):
            print("            ----- event:", e)
            c = [colors.get(i, "purple") for i in clusters[result["event_idx"] == e]]
            model_coords = result["pred"][result["event_idx"] == e]
            if model_coords.shape[1] == 5:
                model_coords = model_coords[:, 1:]
            model_coords = calc_eta_phi(model_coords, 0)
            plot_event(dataset[e], colors=c, ax=ax[e, 2*mn], pfcands=dataset.pfcands_key)
            plot_event(dataset[e], colors=c, ax=ax[e, 2*mn+1], custom_coords=model_coords, pfcands=dataset.pfcands_key)
            plot_event(dataset[e], colors=c, ax=ax1[e, mn], pfcands=dataset.pfcands_key)
            uj = dataset[e].model_jets_unfiltered
            # print the pt of the jet in the middle of each cluster with font size 12
            for i in range(len(uj.pt)):
                ax[e, 2*mn].text(uj.eta[i], uj.phi[i], round(uj.pt[i].item(), 1), color="black", fontsize=10, alpha=0.5)
                ax1[e, mn].text(uj.eta[i], uj.phi[i], round(uj.pt[i].item(), 1), color="black", fontsize=10, alpha=0.5)
                #ax[e, 2*mn+1].text(model_coords[0][i], model_coords[1][i], round(uj.pt[i].item(), 1), color="black", fontsize=10, alpha=0.5)
            ax[e, 2*mn].set_title(model)
            ax1[e, mn].set_title(model)
            ax[e, 2*mn + 1].set_title(model + " (virt. coord.)")
        fig.tight_layout()
        fig1.tight_layout()
        fname = os.path.join(output_path, f"noDM_m_med_{m_med}_m_dark_{m_dark}_r_inv_{str(r_inv).replace('.','')}.pdf")
        fig.savefig(fname)
        fig1.savefig(os.path.join(output_path, f"noDM_m_med_{m_med}_m_dark_{m_dark}_r_inv_{str(r_inv).replace('.','')}_real_only.pdf"))
        print("Saving to", fname)
