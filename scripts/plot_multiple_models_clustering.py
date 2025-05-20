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
    if "qcd" in name.lower():
        return 0, 0, 0 # Standard Model events
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
import fastjet
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
            name = "sc. "
            if run.config["augment_soft_particles"]:
                name += " (aug)"
            if run.config["gt_radius"]:
                name += " GT_R=" + str(run.config["gt_radius"])
            if "transformer" in run.config["network_config"]:
                name += " (T)"
        if run.config["load_from_run"] == "debug_IRC_loss_weighted100_plus_ghosts_2025_04_09_13_48_55_569":
            name += " IRC"
        elif run.config["load_from_run"] == "LGATr_500part_NOQMin_2025_04_09_21_53_37_210":
            name += " NoIRC"
        elif run.config["load_from_run"] == "IRC_loss_Split_and_Noise_alternate_NoAug_2025_04_11_16_15_48_955":
            name += " IRC S+N"
        models[name] = "train/" + run.name
    return models

# with pt=1e-2 ghost particles, also trained on this

#models = get_models_from_tag("eval_19March2025_small_aug_vanishing_momentum_Qcap05_p1e-2")

#models = get_models_from_tag("eval_19March2025_small_aug_vanishing_momentum")
#models = get_models_from_tag("SmallDSReprod2")
#models  = get_models_from_tag("eval_19March2025_pt1e-2_500particles_NoQMinReprod")

'''
models = {}
#models["PL_aug_working"] = "train/Eval_eval_19March2025_small_aug_FTsoft1_2025_03_27_17_15_24_17" # This one was working ~ok for parton-level, why doesn't it work anymore?
models["reprod1"] = "train/Eval_eval_19March2025_small_aug_vanishing_momentum_Qcap05_p1e-2_reprod_1_2025_03_30_16_20_37_779" # reprod1 is using the same model as above, but eval'd on pt=1e-2 particles
# reprod2 has pt uniform 0.01-50 particles
models["reprod2"] = "train/Eval_eval_19March2025_reprod_2_2025_03_30_17_37_54_193"
# reprod3: hdbscan min_samples set to 0

'''

models = {
    "L-GATr": "train/Eval_DelphesPFfix_2025_05_05_08_21_23_380"
}

models = {
    "L-GATr": "train/Eval_DelphesPFfix_FullDataset_QCD_2025_05_15_17_42_39_541"
}

models = {
    "QCD": "train/Eval_DelphesPFfix_FullDataset_TrainDSstudy_QCD_2025_05_18_21_54_43_705",
    "700_07+900_03+QCD": "train/Eval_DelphesPFfix_FullDataset_TrainDSstudy_QCD_2025_05_18_22_18_36_991"
}

print(models)

# R = 2.0 models
#models = {
#    "parton-level": "train/Eval_eval_19March2025_2025_03_19_22_55_48",
#    "gen-level": "train/Eval_eval_19March2025_2025_03_19_23_20_01",
#    "scouting PFCands": "train/Eval_eval_19March2025_2025_03_19_23_4x3_07"
#}


output_path = get_path("QCD_plots_20052025_eval_QCD_train_s_vs_sb", "results")
Path(output_path).mkdir(parents=1, exist_ok=1)

sz = 3
n_events_per_file = 50
# len(models) columns, n_events_per_file rows
from src.layers.object_cond import calc_eta_phi

for ds in range(25):
    print("-------- DS:", ds)
    fig, ax = plt.subplots(n_events_per_file, len(models) * 3, # Colored by the model clusters,
                           figsize=(len(models) * sz * 3, n_events_per_file * sz))
    # also one only with real coordinates
    fig1, ax1 = plt.subplots(n_events_per_file, len(models)+1,
                            figsize=(len(models) * sz, n_events_per_file * sz))
    for mn, model in enumerate(sorted(models.keys())):
        print("    -------- Model:", model)
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
        run_config = get_run_by_name(dataset_path.split("/")[-1]).config
        dataset = EventDataset.from_directory(result["filename"], mmap=True, model_output_file=filename,
                                              model_clusters_file=clusters_file, include_model_jets_unfiltered=True,
                                              aug_soft=run_config["augment_soft_particles"], seed=1000000,
                                              parton_level=run_config["parton_level"],
                                              gen_level=run_config["gen_level"], fastjet_R=[0.8])
        for e in range(n_events_per_file):
            print("            ----- event:", e)
            uj = dataset[e].model_jets_unfiltered
            fj_jets, assignment = EventDataset.get_fastjet_jets_with_assignment(dataset[e], fastjet.JetDefinition(fastjet.antikt_algorithm, 0.8),
                                                                "pfcands", pt_cutoff=30)
            cl = clusters[result["event_idx"] == e]
            large_pt_clusters = []
            for i in np.unique(cl):
                if i == -1: continue
                if uj.pt[i].item() >= 30:
                    large_pt_clusters.append(i)
            #c = [colors.get(i, "purple") for i in clusters[result["event_idx"] == e]]
            c_ak = []
            c = []
            print("Large pt clusters:", large_pt_clusters)
            for i in range(len(cl)):
                if i not in assignment:
                    c_ak.append("purple")
                else:
                    c_ak.append(colors.get(assignment[i], "purple"))

            for i in clusters[result["event_idx"] == e]:
                if i in large_pt_clusters:
                    c.append(colors.get(large_pt_clusters.index(i), "purple"))
                else:
                    c.append("purple")
            model_coords = result["pred"][result["event_idx"] == e]
            if model_coords.shape[1] == 5:
                model_coords = model_coords[:, 1:]
            model_coords = calc_eta_phi(model_coords, 0)
            plot_event(dataset[e], colors=c, ax=ax[e, 3*mn], pfcands=dataset.pfcands_key)
            plot_event(dataset[e], colors=c, ax=ax[e, 3*mn+2], custom_coords=model_coords, pfcands=dataset.pfcands_key)
            plot_event(dataset[e], colors=c_ak, ax=ax[e, 3*mn+1], pfcands=dataset.pfcands_key)
            plot_event(dataset[e], colors=c, ax=ax1[e, mn], pfcands=dataset.pfcands_key)

            # print the pt of the jet in the middle of each cluster with font size 12
            for j in range(len(fj_jets)):
                if fj_jets.pt[j].item() >= 30:
                    ax[e, 3*mn].text(fj_jets.eta[j].item()+0.1, fj_jets.phi[j].item()+0.1, "AK pt="+str(round(fj_jets.pt[j].item(), 1)), color="blue", fontsize=6, alpha=0.5)
            for i in range(len(uj.pt)):
                if uj.pt[i].item() >= 30:
                    ax[e, 3*mn].text(uj.eta[i], uj.phi[i], "M pt=" + str(round(uj.pt[i].item(), 1)), color="black", fontsize=6, alpha=0.5)
                    ax1[e, mn].text(uj.eta[i], uj.phi[i], "M pt=" + str(round(uj.pt[i].item(), 1)), color="black", fontsize=6, alpha=0.5)
                #ax[e, 2*mn+1].text(model_coords[0][i], model_coords[1][i], round(uj.pt[i].item(), 1), color="black", fontsize=10, alpha=0.5)
            ax[e, 3 * mn].set_title(model)
            ax1[e, mn].set_title(model)
            ax[e, 3 * mn + 2].set_title(model + " (clust. space)")
            ax[e, 3 * mn + 1].set_title(model + " (colored AK clust.)")
        fig.tight_layout()
        fig1.tight_layout()
        fname = os.path.join(output_path, f"m_med_{m_med}_m_dark_{m_dark}_r_inv_{str(r_inv).replace('.','')}.pdf")
        fig.savefig(fname)
        fig1.savefig(os.path.join(output_path, f"m_med_{m_med}_m_dark_{m_dark}_r_inv_{str(r_inv).replace('.','')}_real_only.pdf"))
        print("Saving to", fname)
