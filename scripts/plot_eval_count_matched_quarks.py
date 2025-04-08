import os
from tqdm import tqdm
import argparse
import pickle
from src.plotting.eval_matrix import matrix_plot, scatter_plot
from src.utils.paths import get_path
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=False, default="scouting_PFNano_signals2/SVJ_hadronic_std/batch_eval/objectness_score")
parser.add_argument("--threshold-obj-score", "-os-threshold", type=float, default=-1)

thresholds = np.linspace(0.1, 1, 20)
# also add 100 points between 0 and 0.1 at the beginning
thresholds = np.concatenate([np.linspace(0, 0.1, 100), thresholds])

args = parser.parse_args()
path = get_path(args.input, "results")

models = sorted([x for x in os.listdir(path) if not os.path.isfile(os.path.join(path, x))])
models = [x for x in models if "AKX" not in x]
print("Models:", models)

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
    "LGATr_R20": 2.0,
    "LGATr_R25": 2.5
}

comments = {
    "Eval_params_study_2025_02_17_13_30_50": ", tr. on 07_700",
    "Eval_objectness_score_2025_02_12_15_34_33": ", tr. on 03_900, GT=all",
    "Eval_objectness_score_2025_02_18_08_48_13": ", tr. on 03_900, GT=closest",
    "Eval_objectness_score_2025_02_14_11_10_14": ", tr. on 03_900, GT=closest",
    "Eval_objectness_score_2025_02_21_14_51_07": ", tr. on 07_700",
    "Eval_objectness_score_2025_02_10_14_59_49": ", tr. on 03_900, GT=all",
    "Eval_objectness_score_2025_02_23_19_26_25": ", tr. on all, GT=closest",
    "Eval_objectness_score_2025_02_23_21_04_33": ", tr. on 03_900, GT=closest"
}

out_file_PR = os.path.join(get_path(args.input, "results"), "precision_recall.pdf")
out_file_PG = os.path.join(get_path(args.input, "results"), "PLoverGL.pdf")

if args.threshold_obj_score != -1:
    out_file_PR_OS = os.path.join(get_path(args.input, "results"), f"precision_recall_with_obj_score.pdf")

out_file_avg_number_matched_quarks = os.path.join(get_path(args.input, "results"), "avg_number_matched_quarks.pdf")

def get_plots_for_params(mMed, mDark, rInv, result_PR_thresholds):
    precisions = []
    recalls = []
    f1_scores = []
    for i in range(len(thresholds)):
        if result_PR_thresholds[mMed][mDark][rInv][i][1] == 0:
            precisions.append(0)
        else:
            precisions.append(
                result_PR_thresholds[mMed][mDark][rInv][i][0] / result_PR_thresholds[mMed][mDark][rInv][i][1])
        if result_PR_thresholds[mMed][mDark][rInv][i][2] == 0:
            recalls.append(0)
        else:
            recalls.append(
                result_PR_thresholds[mMed][mDark][rInv][i][0] / result_PR_thresholds[mMed][mDark][rInv][i][2])
    for i in range(len(thresholds)):
        if precisions[i] + recalls[i] == 0:
            f1_scores.append(0)
        else:
            f1_scores.append(2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i]))
    return precisions, recalls, f1_scores

sz = 5
nplots = 9

# Now make 3 plots, one for mMed=700,r_inv=0.7; one for mMed=700,r_inv=0.5; one for mMed=700,r_inv=0.3
###fig, ax = plt.subplots(3, 3, figsize=(3 * sz, 3 * sz))

'''fig, ax = plt.subplots(3, nplots, figsize=(nplots*sz, 3*sz))
for mi, mass in enumerate([700, 900, 1500]):
    start_idx = mi*3
    for i0, rinv in enumerate([0.3, 0.5, 0.7]):
        i = start_idx + i0
        # 0 is precision, 1 is recall, 2 is f1 score
        ax[0, i].set_title(f"r_inv={rinv}, m_med={mass} GeV")
        ax[1, i].set_title(f"r_inv={rinv}, m_med={mass} GeV")
        ax[2, i].set_title(f"r_inv={rinv}, m_med={mass} GeV")
        ax[0, i].set_ylabel("Precision")
        ax[1, i].set_ylabel("Recall")
        ax[2, i].set_ylabel("F1 score")
        ax[0, i].grid()
        ax[1, i].grid()
        ax[2, i].grid()
        ylims = {} # key: j and i
        default_ylims = [1, 0]
        for j, model in enumerate(models):
            result_PR_thresholds = os.path.join(path, model, "count_matched_quarks", "result_PR_thresholds.pkl")
            if not os.path.exists(result_PR_thresholds):
                continue
            run_config = pickle.load(open(os.path.join(path, model, "run_config.pkl"), "rb"))
            result_PR_thresholds = pickle.load(open(result_PR_thresholds, "rb"))
            if mass not in result_PR_thresholds:
                continue
            if rinv not in result_PR_thresholds[mass]:
                continue
            precisions, recalls, f1_scores = get_plots_for_params(mass, 20, rinv, result_PR_thresholds)
            if not run_config["gt_radius"] == 0.8:
                continue
            label = "R={} gl.f.={} {}".format(run_config["gt_radius"], run_config.get("global_features_obj_score", False), comments.get(run_config["run_name"], run_config["run_name"]))
            scatter_plot(ax[0, i], thresholds, precisions, label=label)
            scatter_plot(ax[1, i], thresholds, recalls, label=label)
            scatter_plot(ax[2, i], thresholds, f1_scores, label=label)
            #ylims[0] = [min(ylims[0][0], min(precisions)), max(ylims[0][1], max(precisions))]
            #ylims[1] = [min(ylims[1][0], min(recalls)), max(ylims[1][1], max(recalls))]
            #ylims[2] = [min(ylims[2][0], min(f1_scores)), max(ylims[2][1], max(f1_scores))]
            filt = thresholds < 0.2
            precisions = np.array(precisions)[filt]
            recalls = np.array(recalls)[filt]
            f1_scores = np.array(f1_scores)[filt]
            if (i, 0) not in ylims:
                ylims[(i, 0)] = default_ylims
            upper_factor = 1.01
            lower_factor = 0.99
            ylims[(i, 0)] = [min(ylims[(i, 0)][0], min(precisions)*lower_factor), max(ylims[(i, 0)][1], max(precisions)*upper_factor)]
            if (i, 1) not in ylims:
                ylims[(i, 1)] = default_ylims
            ylims[(i, 1)] = [min(ylims[(i, 1)][0], min(recalls)*lower_factor), max(ylims[(i, 1)][1], max(recalls)*upper_factor)]
            if (i, 2) not in ylims:
                ylims[(i, 2)] = default_ylims
            ylims[(i, 2)] = [min(ylims[(i, 2)][0], min(f1_scores)*lower_factor), max(ylims[(i, 2)][1], max(f1_scores)*upper_factor)]
        for j in range(3):
            ax[j, i].set_ylim(ylims[(i, j)])
            ax[j, i].legend()
            ax[j, i].set_xlim([0, 0.2])
            ax[j, i].set_xlim([0, 0.2])
            ax[j, i].set_xlim([0, 0.2])
        # now adjust the ylim so that the plots are more readable

fig.tight_layout()
fig.savefig(os.path.join(get_path(args.input, "results"), "precision_recall_thresholds.pdf"))

print("Saved to", os.path.join(get_path(args.input, "results"), "precision_recall_thresholds.pdf"))'''


import wandb
api = wandb.Api()

def get_run_by_name(name):
    clust_suffix = ""
    if name.endswith("FT"):
        #remove FT from the end
        name = name[:-2]
        clust_suffix = "FT"
    if name.endswith("FT1"):
        #remove FT from the end # min-samples 1 min-cluster-size 2 epsilon 0.3
        name = name[:-3]
        clust_suffix = "FT1"
    if name.endswith("10_5"):
        name = name[:-4]
        clust_suffix = "10_5"
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
    return runs[0], clust_suffix


def get_run_config(run_name):
    r, clust_suffix = get_run_by_name(run_name)
    if r is None:
        print("Getting info from run", run_name, "failed")
        return None, None
    config = r.config
    result = {}
    if config["parton_level"]:
        prefix = "parton level"
        result["level"] = "PL"
    elif config["gen_level"]:
        prefix = "gen level"
        result["level"] = "GL"
    else:
        prefix = "scouting"
        result["level"] = "scouting"
    if config["augment_soft_particles"]:
        result["ghosts"] = True
        result["level"] += "+ghosts"
    gt_r = config["gt_radius"]
    if config.get("augment_soft_particles", False):
        prefix += " (aug)" # ["LGATr_training_NoPID_10_16_64_0.8_Aug_Finetune_vanishing_momentum_QCap05_2025_03_28_17_12_25_820", "LGATr_training_NoPID_10_16_64_2.0_Aug_Finetune_vanishing_momentum_QCap05_2025_03_28_17_12_26_400"]
    training_datasets = {
        "LGATr_training_NoPID_10_16_64_0.8_AllData_2025_02_28_13_42_59": "all",
        "LGATr_training_NoPID_10_16_64_0.8_2025_02_28_12_42_59": "900_03",
        "LGATr_training_NoPID_10_16_64_2.0_2025_02_28_12_48_58": "900_03",
        "LGATr_training_NoPID_10_16_64_0.8_700_07_2025_02_28_13_01_59": "700_07",
        "LGATr_training_NoPIDGL_10_16_64_0.8_2025_03_17_20_05_04": "900_03_GenLevel",
        "LGATr_training_NoPIDGL_10_16_64_2.0_2025_03_17_20_05_04": "900_03_GenLevel",
        "Transformer_training_NoPID_10_16_64_2.0_2025_03_03_17_00_38": "900_03_T",
        "Transformer_training_NoPID_10_16_64_0.8_2025_03_03_15_55_50": "900_03_T",
        "LGATr_training_NoPID_10_16_64_0.8_Aug_Finetune_2025_03_27_12_46_12_740": "900_03+SoftAug",
        "LGATr_training_NoPID_10_16_64_2.0_Aug_Finetune_vanishing_momentum_2025_03_28_10_43_36_81": "900_03+SoftAugVM",
        "LGATr_training_NoPID_10_16_64_0.8_Aug_Finetune_vanishing_momentum_2025_03_28_10_43_37_44": "900_03+SoftAugVM",
        "LGATr_training_NoPID_10_16_64_0.8_Aug_Finetune_vanishing_momentum_QCap05_2025_03_28_17_12_25_820": "900_03+qcap05",
        "LGATr_training_NoPID_10_16_64_2.0_Aug_Finetune_vanishing_momentum_QCap05_2025_03_28_17_12_26_400": "900_03+qcap05",
        "LGATr_training_NoPID_10_16_64_2.0_Aug_Finetune_vanishing_momentum_QCap05_1e-2_2025_03_29_14_58_38_650": "pt 1e-2",
        "LGATr_training_NoPID_10_16_64_0.8_Aug_Finetune_vanishing_momentum_QCap05_1e-2_2025_03_29_14_58_36_446": "pt 1e-2",
        "LGATr_pt_1e-2_500part_2025_04_01_16_49_08_406": "500_pt_1e-2_PLFT",
        "LGATr_pt_1e-2_500part_2025_04_01_21_14_07_350": "500_pt_1e-2_PLFT",
        "LGATr_pt_1e-2_500part_NoQMin_2025_04_03_23_15_17_745": "500_1e-2_scFT",
        "LGATr_pt_1e-2_500part_NoQMin_2025_04_03_23_15_35_810": "500_1e-2_scFT",
        "LGATr_pt_1e-2_500part_NoQMin_10_to_1000p_2025_04_04_12_57_51_536": "10_1000_1e-2_scFT",
        "LGATr_pt_1e-2_500part_NoQMin_10_to_1000p_2025_04_04_12_57_47_788": "10_1000_1e-2_scFT",
        "LGATr_pt_1e-2_500part_NoQMin_10_to_1000p_CW0_2025_04_04_15_30_16_839": "10_1000_1e-2_CW0",
        "LGATr_pt_1e-2_500part_NoQMin_10_to_1000p_CW0_2025_04_04_15_30_20_113": "10_1000_1e-2_CW0"
    }

    train_name = config["load_from_run"]
    ckpt_step = config["ckpt_step"]
    print("train name", train_name)
    if train_name not in training_datasets:
        print("!! unknown run", train_name)
    training_dataset = training_datasets.get(train_name, train_name) + "_s" + str(ckpt_step) + clust_suffix
    if "plptfilt01" in run_name.lower():
        training_dataset += "_PLPtFiltMinPt01" # min pt 0.1
    elif "noplfilter" in run_name.lower():
        training_dataset += "_noPLFilter"
    elif "noplptfilter" in run_name.lower():
        training_dataset += "_noPLPtFilter" # actually there was a 0.5 pt cut in the ntuplizer, removed by plptfilt01
    elif "nopletafilter" in run_name.lower():
        training_dataset += "_noPLEtaFilter"
    result["GT_R"] = gt_r
    result["training_dataset"] = training_dataset
    return f"GT_R={gt_r}, tr.: {training_dataset}, {prefix}", result


sz = 5
ak_path = os.path.join(path, "AKX_PL", "count_matched_quarks")
result_PR_AKX = pickle.load(open(os.path.join(ak_path, "result_PR_AKX.pkl"), "rb"))
radius = [0.8, 2.0]
def select_radius(d, radius, depth=3):
    # from the dictionary, select radius at the level
    if depth == 0:
        return d[radius]
    return {key: select_radius(d[key], radius, depth - 1) for key in d}

if len(models):
    fig, ax = plt.subplots(3, len(models) + len(radius), figsize=(sz * len(models), sz * 3))
    for i, model in tqdm(enumerate(sorted(models))):
        output_path = os.path.join(path, model, "count_matched_quarks")
        if not os.path.exists(os.path.join(output_path, "result.pkl")):
            continue
        result = pickle.load(open(os.path.join(output_path, "result.pkl"), "rb"))
        #result_unmatched = pickle.load(open(os.path.join(output_path, "result_unmatched.pkl"), "rb"))
        #result_fakes = pickle.load(open(os.path.join(output_path, "result_fakes.pkl"), "rb"))
        result_bc = pickle.load(open(os.path.join(output_path, "result_bc.pkl"), "rb"))
        result_PR = pickle.load(open(os.path.join(output_path, "result_PR.pkl"), "rb"))
        #matrix_plot(result, "Blues", "Avg. matched dark quarks / event").savefig(os.path.join(output_path, "avg_matched_dark_quarks.pdf"), ax=ax[0, i])
        #matrix_plot(result_fakes, "Greens", "Avg. unmatched jets / event").savefig(os.path.join(output_path, "avg_unmatched_jets.pdf"), ax=ax[1, i])
        matrix_plot(result_PR, "Oranges", "Precision (N matched dark quarks / N predicted jets)", metric_comp_func = lambda r: r[0], ax=ax[0, i])
        matrix_plot(result_PR, "Reds", "Recall (N matched dark quarks / N dark quarks)", metric_comp_func = lambda r: r[1], ax=ax[1, i])
        matrix_plot(result_PR, "Purples", r"$F_1$ score", metric_comp_func = lambda r: 2 * r[0] * r[1] / (r[0] + r[1]), ax=ax[2, i])
        run_config_title, run_config = get_run_config(model)
        print("RC title", run_config_title)
        if run_config is None:
            print("Skipping", model)
            continue
        ax[0, i].set_title(run_config_title)
        ax[1, i].set_title(run_config_title)
        ax[2, i].set_title(run_config_title)
        print(model, run_config_title)
    for i, R in enumerate(radius):
        result_PR_AKX_current = select_radius(result_PR_AKX, R)
        matrix_plot(result_PR_AKX_current, "Oranges", "Precision (N matched dark quarks / N predicted jets)",
                    metric_comp_func=lambda r: r[0], ax=ax[0, i+len(models)])
        matrix_plot(result_PR_AKX_current, "Reds", "Recall (N matched dark quarks / N dark quarks)",
                    metric_comp_func=lambda r: r[1], ax=ax[1, i+len(models)])
        matrix_plot(result_PR_AKX_current, "Purples", r"$F_1$ score", metric_comp_func=lambda r: 2 * r[0] * r[1] / (r[0] + r[1]),
                    ax=ax[2, i+len(models)])
        ax[0, i+len(models)].set_title(f"AK, R={R}")
        ax[1, i+len(models)].set_title(f"AK, R={R}")
        ax[2, i+len(models)].set_title(f"AK, R={R}")

    fig.tight_layout()
    fig.savefig(out_file_PR)
    print("Saved to", out_file_PR)

## Now do the GT R vs metrics plots

oranges = plt.get_cmap("Oranges")
reds = plt.get_cmap("Reds")
purples = plt.get_cmap("Purples")

mDark = 20
to_plot = {} # training dataset -> rInv -> mMed -> level -> "f1score" -> value
results_all = {}
results_all_ak = {}
plotting_hypotheses = [[700, 0.7], [700, 0.5], [700, 0.3], [900, 0.3], [900, 0.7]]
sz_small = 5
for j, model in enumerate(models):
    _, rc = get_run_config(model)
    if rc is None or model in ["Eval_eval_19March2025_pt1e-2_500particles_FT_PL_2025_04_02_14_28_33_421FT", "Eval_eval_19March2025_pt1e-2_500particles_FT_PL_2025_04_02_14_47_23_671FT", "Eval_eval_19March2025_small_aug_vanishing_momentum_2025_03_28_11_45_16_582", "Eval_eval_19March2025_small_aug_vanishing_momentum_2025_03_28_11_46_26_326"]:
        print("Skipping", model)
        continue
    td = rc["training_dataset"]
    level = rc["level"]
    r = rc["GT_R"]
    output_path = os.path.join(path, model, "count_matched_quarks")
    if not os.path.exists(os.path.join(output_path, "result_PR.pkl")):
        continue
    result_PR = pickle.load(open(os.path.join(output_path, "result_PR.pkl"), "rb"))
    print(level)
    if td not in to_plot:
        to_plot[td] = {}
        results_all[td] = {}
    for mMed_h in result_PR:
        if mMed_h not in results_all[td]:
            results_all[td][mMed_h] = {20: {}}
        for rInv_h in result_PR[mMed_h][20]:
            if rInv_h not in results_all[td][mMed_h][20]:
                results_all[td][mMed_h][20][rInv_h] = {}
            #for level in ["PL+ghosts", "GL+ghosts", "scouting+ghosts"]:
            if level not in results_all[td][mMed_h][20][rInv_h]:
                results_all[td][mMed_h][20][rInv_h][level] = {}
            if r not in results_all[td][mMed_h][20][rInv_h][level]:
                precision = result_PR[mMed_h][mDark][rInv_h][0]
                recall = result_PR[mMed_h][mDark][rInv_h][1]
                f1score = 2 * precision * recall / (precision + recall)
                results_all[td][mMed_h][20][rInv_h][level][r] = f1score
    for i, h in enumerate(plotting_hypotheses):
        mMed_h, rInv_h = h
        if rInv_h not in to_plot[td]:
            to_plot[td][rInv_h] = {}
        print("Model", model)
        if mMed_h not in to_plot[td][rInv_h]:
            to_plot[td][rInv_h][mMed_h] = {} # level

        if level not in to_plot[td][rInv_h][mMed_h]:
            to_plot[td][rInv_h][mMed_h][level] = {"precision": [], "recall": [], "f1score": [], "R": []}
        precision = result_PR[mMed_h][mDark][rInv_h][0]
        recall = result_PR[mMed_h][mDark][rInv_h][1]
        f1score = 2 * precision * recall / (precision + recall)
        to_plot[td][rInv_h][mMed_h][level]["precision"].append(precision)
        to_plot[td][rInv_h][mMed_h][level]["recall"].append(recall)
        to_plot[td][rInv_h][mMed_h][level]["f1score"].append(f1score)
        to_plot[td][rInv_h][mMed_h][level]["R"].append(r)


to_plot_ak = {} # level ("scouting"/"GL"/"PL") -> rInv -> mMed -> {"f1score": [], "R": []}

for j, model in enumerate(["AKX", "AKX_PL", "AKX_GL"]):
    print(model)
    if os.path.exists(os.path.join(path, model, "count_matched_quarks", "result_PR_AKX.pkl")):
        result_PR_AKX = pickle.load(open(os.path.join(path, model, "count_matched_quarks", "result_PR_AKX.pkl"), "rb"))
    else:
        print("Skipping", model)
        continue
    level = "scouting"
    if "PL" in model:
        level = "PL"
    elif "GL" in model:
        level = "GL"
    if level not in to_plot_ak:
        to_plot_ak[level] = {}
    for mMed_h in result_PR_AKX:
        if mMed_h not in results_all_ak:
            results_all_ak[mMed_h] = {20: {}}
        for rInv_h in result_PR_AKX[mMed_h][20]:
            if rInv_h not in results_all_ak[mMed_h][20]:
                results_all_ak[mMed_h][20][rInv_h] = {}
            if level not in results_all_ak[mMed_h][20][rInv_h]:
                results_all_ak[mMed_h][20][rInv_h][level] = {}
            for ridx, R in enumerate(result_PR_AKX[mMed_h][20][rInv_h]):
                if R not in results_all_ak[mMed_h][20][rInv_h][level]:
                    precision = result_PR_AKX[mMed_h][mDark][rInv_h][R][0]
                    recall = result_PR_AKX[mMed_h][mDark][rInv_h][R][1]
                    f1score = 2 * precision * recall / (precision + recall)
                    results_all_ak[mMed_h][20][rInv_h][level][R] = f1score

    for i, h in enumerate(plotting_hypotheses):
        mMed_h, rInv_h = h
        if rInv_h not in to_plot_ak[level]:
            to_plot_ak[level][rInv_h] = {}
        print("Model", model)
        rs = sorted(result_PR_AKX[mMed_h][20][rInv_h].keys())
        if mMed_h not in to_plot_ak[level][rInv_h]:
            to_plot_ak[level][rInv_h][mMed_h] = {"precision": [], "recall": [], "f1score": [], "R": []}
        precision = np.array([result_PR_AKX[mMed_h][mDark][rInv_h][i][0] for i in rs])
        recall = np.array([result_PR_AKX[mMed_h][mDark][rInv_h][i][1] for i in rs])
        f1score = 2 * precision * recall / (precision + recall)
        to_plot_ak[level][rInv_h][mMed_h]["precision"] = precision
        to_plot_ak[level][rInv_h][mMed_h]["recall"] = recall
        to_plot_ak[level][rInv_h][mMed_h]["f1score"] = f1score
        to_plot_ak[level][rInv_h][mMed_h]["R"] = rs
print("AK:", to_plot_ak)
fig, ax = plt.subplots(len(to_plot) + 1, len(plotting_hypotheses), figsize=(sz_small * len(plotting_hypotheses), sz_small * len(to_plot))) # also add AKX as last plot

colors = {
    #"PL": "green",
    #"GL": "blue",
    #"scouting": "red",
    "PL+ghosts": "green",
    "GL+ghosts": "blue",
    "scouting+ghosts": "red"
}
ak_colors = {
    "PL": "green",
    "GL": "blue",
    "scouting": "red",
}
for i, td in enumerate(to_plot):
    # for each training dataset
    for j, h in enumerate(plotting_hypotheses):
        ax[i, j].set_title(f"r_inv={h[1]}, m={h[0]}, tr. on {td}")
        ax[i, j].set_ylabel("F1 score")
        ax[i, j].set_xlabel("GT R")
        ax[i, j].grid()
        for level in sorted(list(to_plot[td][h[1]][h[0]].keys())):
            print("level", level)
            print("Plotting", td, h[1], h[0], level)
            if level in colors:
                ax[i, j].plot(to_plot[td][h[1]][h[0]][level]["R"], to_plot[td][h[1]][h[0]][level]["f1score"], ".-", label=level, color=colors[level])
        ax[i, j].legend()
for j, h in enumerate(plotting_hypotheses): # for to_plot_AK
    ax[-1, j].set_title(f"r_inv={h[1]}, m={h[0]}, AK baseline")
    ax[-1, j].set_ylabel("F1 score")
    ax[-1, j].set_xlabel("GT R")
    ax[-1, j].grid()
    for i, ak_level in enumerate(sorted(list(to_plot_ak.keys()))):
        mMed_h, rInv_h = h
        if ak_level in ak_colors:
            ax[-1, j].plot(to_plot_ak[ak_level][rInv_h][mMed_h]["R"], to_plot_ak[ak_level][rInv_h][mMed_h]["f1score"], ".-", label=ak_level, color=ak_colors[ak_level])
    ax[-1, j].legend()
fig.tight_layout()
fig.savefig(os.path.join(get_path(args.input, "results"), "score_vs_GT_R_plots_1.pdf"))
print("Saved to", os.path.join(get_path(args.input, "results"), "score_vs_GT_R_plots_1.pdf"))

fig, ax = plt.subplots(1, len(results_all)*len(radius) + len(radius), figsize=(7 * len(results_all)*len(radius)+len(radius), 5))
for i, model in enumerate(results_all):
    for j, R in enumerate(radius):
        #if r not in results_all[model][700][20][0.3]["scouting"]:
        #    continue
        # for each training dataset
        index = len(radius)*i + j
        ax[index].set_title(model + " R={}".format(R))
        matrix_plot(results_all[model], "Greens", r"PL/GL F1 score", ax=ax[index], metric_comp_func=lambda r: r["PL+ghosts"][R]/r["scouting+ghosts"][R])
for i, R in enumerate(radius):
    index = len(radius)*len(results_all) + i
    ax[index].set_title("AK R={}".format(R))
    matrix_plot(results_all_ak, "Greens", r"PL/GL F1 score", ax=ax[index], metric_comp_func=lambda r: r["PL"][R]/r["GL"][R])
fig.tight_layout()
fig.savefig(out_file_PG)

print("Saved to", out_file_PG)


1/0


#print("Saved to", os.path.join(get_path(args.input, "results"), "score_vs_GT_R_plots_AK.pdf"))
#print("Saved to", os.path.join(get_path(args.input, "results"), "score_vs_GT_R_plots_AK_ratio.pdf"))

########### Now save the above plot with objectness score applied

if args.threshold_obj_score != -1:
    fig, ax = plt.subplots(3, len(models), figsize=(sz * len(models), sz * 3))
    for i, model in tqdm(enumerate(models)):
        output_path = os.path.join(path, model, "count_matched_quarks")
        if not os.path.exists(os.path.join(output_path, "result.pkl")):
            continue
        result = pickle.load(open(os.path.join(output_path, "result.pkl"), "rb"))
        #result_unmatched = pickle.load(open(os.path.join(output_path, "result_unmatched.pkl"), "rb"))
        result_fakes = pickle.load(open(os.path.join(output_path, "result_fakes.pkl"), "rb"))
        result_bc = pickle.load(open(os.path.join(output_path, "result_bc.pkl"), "rb"))
        result_PR = pickle.load(open(os.path.join(output_path, "result_PR.pkl"), "rb"))
        result_PR_thresholds = pickle.load(open(os.path.join(output_path, "result_PR_thresholds.pkl"), "rb"))
        #thresholds = sorted(list(result_PR_thresholds[900][20][0.3].keys()))
        #thresholds = np.array(thresholds)
        # now linearly interpolate the thresholds and set the j according to args.threshold_obj_score
        j = np.argmin(np.abs(thresholds - args.threshold_obj_score))
        print("Thresholds", thresholds)
        print("Chose j=", j, "for threshold", args.threshold_obj_score, "(effectively it's", thresholds[j], ")")
        def wrap(r):
            # compute [precision, recall] array from [n_relevant_retrieved, all_retrieved, all_relevant]
            if r[1] == 0 or r[2] == 0:
                return [0, 0]
            return [r[0] / r[1], r[0] / r[2]]
        matrix_plot(result_PR_thresholds, "Oranges", "Precision (N matched dark quarks / N predicted jets)", metric_comp_func = lambda r: wrap(r[j])[0], ax=ax[0, i])
        matrix_plot(result_PR_thresholds, "Reds", "Recall (N matched dark quarks / N dark quarks)", metric_comp_func = lambda r: wrap(r[j])[1], ax=ax[1, i])
        matrix_plot(result_PR_thresholds, "Purples", r"$F_1$ score", metric_comp_func = lambda r: 2 * wrap(r[j])[0] * wrap(r[j])[1] / (wrap(r[j])[0] + wrap(r[j])[1]), ax=ax[2, i])
        ax[0, i].set_title(model)
        ax[1, i].set_title(model)
        ax[2, i].set_title(model)

    fig.tight_layout()
    fig.savefig(out_file_PR_OS)
    print("Saved to", out_file_PR_OS)

################
# UNUSED PLOTS #
################
'''fig, ax = plt.subplots(2, len(models), figsize=(sz * len(models), sz * 2))
for i, model in tqdm(enumerate(models)):
    output_path = os.path.join(path, model, "count_matched_quarks")
    if not os.path.exists(os.path.join(output_path, "result.pkl")):
        continue
    result = pickle.load(open(os.path.join(output_path, "result.pkl"), "rb"))
    #result_unmatched = pickle.load(open(os.path.join(output_path, "result_unmatched.pkl"), "rb"))
    result_fakes = pickle.load(open(os.path.join(output_path, "result_fakes.pkl"), "rb"))
    result_bc = pickle.load(open(os.path.join(output_path, "result_bc.pkl"), "rb"))
    result_PR = pickle.load(open(os.path.join(output_path, "result_PR.pkl"), "rb"))
    matrix_plot(result, "Blues", "Avg. matched dark quarks / event", ax=ax[0, i])
    matrix_plot(result_fakes, "Greens", "Avg. unmatched jets / event", ax=ax[1, i])
    ax[0, i].set_title(model)
    ax[1, i].set_title(model)
fig.tight_layout()
fig.savefig(out_file_avg_number_matched_quarks)
print("Saved to", out_file_avg_number_matched_quarks)'''

rinvs = [0.3, 0.5, 0.7]
sz = 4
fig, ax = plt.subplots(len(rinvs), 3, figsize=(3*sz, sz*len(rinvs)))
fig_AK, ax_AK = plt.subplots(len(rinvs), 3, figsize=(3*sz, sz*len(rinvs)))
fig_AK_ratio, ax_AK_ratio = plt.subplots(len(rinvs), 3, figsize=(3*sz, sz*len(rinvs)))


to_plot = {} # r_inv -> m_med -> precision, recall, R
to_plot_ak = {} # plotting for the AK baseline

### Plotting the score vs GT R plots

oranges = plt.get_cmap("Oranges")
reds = plt.get_cmap("Reds") # Plot a plot for each mass at given r_inv of the precision, recall, F1 score
purples = plt.get_cmap("Purples")

mDark = 20

for i, rinv in enumerate(rinvs):
    if rinv not in to_plot:
        to_plot[rinv] = {}
        to_plot_ak[rinv] = {}
    for j, model in enumerate(models):
        print("Model", model)
        if model not in radius:
            continue
        r = radius[model]
        output_path = os.path.join(path, model, "count_matched_quarks")
        if not os.path.exists(os.path.join(output_path, "result_PR.pkl")):
            continue
        result_PR = pickle.load(open(os.path.join(output_path, "result_PR.pkl"), "rb"))
        #if radius not in to_plot[rinv]:
        #    to_plot[rinv][radius] = {}
        for k, mMed in enumerate(sorted(result_PR.keys())):
            if mMed not in to_plot[rinv]:
                to_plot[rinv][mMed] = {"precision": [], "recall": [], "f1score": [], "R": []}
            precision = result_PR[mMed][mDark][rinv][0]
            recall = result_PR[mMed][mDark][rinv][1]
            f1score = 2 * precision * recall / (precision + recall)
            to_plot[rinv][mMed]["precision"].append(precision)
            to_plot[rinv][mMed]["recall"].append(recall)
            to_plot[rinv][mMed]["f1score"].append(f1score)
            to_plot[rinv][mMed]["R"].append(r)
    for mMed in sorted(to_plot[rinv].keys()):
        # normalize mmed between 0 and 1 (originally between 700 and 3000)
        mmed = (mMed - 500) / (3000 - 500)
        r = to_plot[rinv][mMed]
        print("Model R", r["R"])
        scatter_plot(ax[0, i], r["R"], r["precision"], label="m={} GeV".format(round(mMed)), color=oranges(mmed))
        scatter_plot(ax[1, i], r["R"], r["recall"], label="m={} GeV".format(round(mMed)), color=reds(mmed))
        scatter_plot(ax[2, i], r["R"], r["f1score"], label="m={} GeV".format(round(mMed)), color=purples(mmed))
    if not os.path.exists(os.path.join(ak_path, "result_PR_AKX.pkl")):
        continue
    result_PR_AKX = pickle.load(open(os.path.join(ak_path, "result_PR_AKX.pkl"), "rb"))
    #if radius not in to_plot[rinv]:
    #    to_plot[rinv][radius] = {}
    for k, mMed in enumerate(sorted(result_PR_AKX.keys())):
        if mMed not in to_plot_ak[rinv]:
            to_plot_ak[rinv][mMed] = {"precision": [], "recall": [], "f1score": [], "R": []}
        rs = sorted(result_PR_AKX[mMed][mDark][rinv].keys())
        precision = np.array([result_PR_AKX[mMed][mDark][rinv][k][0] for k in rs])
        recall = np.array([result_PR_AKX[mMed][mDark][rinv][k][1] for k in rs])
        f1score = 2 * precision * recall / (precision + recall)
        to_plot_ak[rinv][mMed]["precision"] += list(precision)
        to_plot_ak[rinv][mMed]["recall"] += list(recall)
        to_plot_ak[rinv][mMed]["f1score"] += list(f1score)
        to_plot_ak[rinv][mMed]["R"] += rs

    for mMed in sorted(to_plot_ak[rinv].keys()):
        # Normalize mmed between 0 and 1 (originally between 700 and 3000)
        mmed = (mMed - 500) / (3000 - 500)
        r = to_plot_ak[rinv][mMed]
        r_model = to_plot[rinv][mMed]
        print("AK R", r["R"])
        scatter_plot(ax_AK[0, i], r["R"], r["precision"], label="m={} GeV AK".format(round(mMed)), color=oranges(mmed), pattern=".--")
        scatter_plot(ax_AK[1, i], r["R"], r["recall"], label="m={} GeV AK".format(round(mMed)), color=reds(mmed), pattern=".--")
        scatter_plot(ax_AK[2, i], r["R"], r["f1score"], label="m={} GeV AK".format(round(mMed)), color=purples(mmed), pattern=".--")
        # r["R"] has more points than r_model["R"] - pick those from r["R"] that are in r_model["R"]
        r["R"] = np.array(r["R"])
        r["precision"] = np.array(r["precision"])
        r["recall"] = np.array(r["recall"])
        r["f1score"] = np.array(r["f1score"])
        filt = np.isin(r["R"], r_model["R"])
        r["R"] = r["R"][filt]
        r["precision"] = r["precision"][filt]
        r["recall"] = r["recall"][filt]
        r["f1score"] = r["f1score"][filt]
        scatter_plot(ax_AK_ratio[0, i], r["R"], r["precision"]/np.array(r_model["precision"]), label="m={} GeV AK".format(round(mMed)), color=oranges(mmed), pattern=".--")
        scatter_plot(ax_AK_ratio[1, i], r["R"], r["recall"]/np.array(r_model["recall"]), label="m={} GeV AK".format(round(mMed)), color=reds(mmed), pattern=".--")
        scatter_plot(ax_AK_ratio[2, i], r["R"], r["f1score"]/np.array(r_model["f1score"]), label="m={} GeV AK".format(round(mMed)), color=purples(mmed), pattern=".--")

    for ax1 in [ax, ax_AK, ax_AK_ratio]:
        ax1[0, i].set_title(f"Precision r_inv = {rinv}")
        ax1[1, i].set_title(f"Recall r_inv = {rinv}")
        ax1[2, i].set_title(f"F1 score r_inv = {rinv}")
        ax1[2, i].legend()
        ax1[1, i].legend()
        ax1[0, i].legend()
        ax1[0, i].grid()
        ax1[1, i].grid()
        ax1[2, i].grid()
        ax1[0, i].set_xlabel("GT R")
        ax1[1, i].set_xlabel("GT R")
        ax1[2, i].set_xlabel("GT R")
    ax_AK_ratio[0, i].set_ylabel("Precision (model=1)")
    ax_AK_ratio[1, i].set_ylabel("Recall (model=1)")
    ax_AK_ratio[2, i].set_ylabel("F1 score (model=1)")
fig.tight_layout()
fig_AK.tight_layout()
fig.savefig(os.path.join(get_path(args.input, "results"), "score_vs_GT_R_plots.pdf"))
fig_AK.savefig(os.path.join(get_path(args.input, "results"), "score_vs_GT_R_plots_AK.pdf"))
fig_AK_ratio.tight_layout()
fig_AK_ratio.savefig(os.path.join(get_path(args.input, "results"), "score_vs_GT_R_plots_AK_ratio.pdf"))

print("Saved to", os.path.join(get_path(args.input, "results"), "score_vs_GT_R_plots_AK.pdf"))
print("Saved to", os.path.join(get_path(args.input, "results"), "score_vs_GT_R_plots_AK_ratio.pdf"))
