import os
from tqdm import tqdm
import argparse
import pickle
from src.plotting.eval_matrix import matrix_plot, scatter_plot
from src.utils.paths import get_path
import matplotlib.pyplot as plt
import numpy as np

# This script produces the pt cutoff vs. f1 score

inputs = {
    30: "Delphes_020425_test_PU_PFfix_part0/batch_eval_2k/DelphesPFfix_FullDataset_pt_30.0",
    40: "Delphes_020425_test_PU_PFfix_part0/batch_eval_2k/DelphesPFfix_FullDataset_pt_40.0",
    50: "Delphes_020425_test_PU_PFfix_part0/batch_eval_2k/DelphesPFfix_FullDataset_pt_50.0",
    60: "Delphes_020425_test_PU_PFfix_part0/batch_eval_2k/DelphesPFfix_FullDataset_pt_60.0",
    70: "Delphes_020425_test_PU_PFfix_part0/batch_eval_2k/DelphesPFfix_FullDataset_pt_70.0",
    80: "Delphes_020425_test_PU_PFfix_part0/batch_eval_2k/DelphesPFfix_FullDataset_pt_80.0",
    90: "Delphes_020425_test_PU_PFfix_part0/batch_eval_2k/DelphesPFfix_FullDataset_pt_90.0",
    100: "Delphes_020425_test_PU_PFfix_part0/batch_eval_2k/DelphesPFfix_FullDataset"
}

inputs = {
    30: "Delphes_020425_test_PU_PFfix_part0/batch_eval_2k/DelphesPFfix_FullDataset_TrainDSstudy_pt_30.0",
    40: "Delphes_020425_test_PU_PFfix_part0/batch_eval_2k/DelphesPFfix_FullDataset_TrainDSstudy_pt_40.0",
    50:  "Delphes_020425_test_PU_PFfix_part0/batch_eval_2k/DelphesPFfix_FullDataset_TrainDSstudy_pt_50.0",
    60: "Delphes_020425_test_PU_PFfix_part0/batch_eval_2k/DelphesPFfix_FullDataset_TrainDSstudy_pt_60.0",
    70: "Delphes_020425_test_PU_PFfix_part0/batch_eval_2k/DelphesPFfix_FullDataset_TrainDSstudy_pt_70.0",
    80: "Delphes_020425_test_PU_PFfix_part0/batch_eval_2k/DelphesPFfix_FullDataset_TrainDSstudy_pt_80.0",
    90: "Delphes_020425_test_PU_PFfix_part0/batch_eval_2k/DelphesPFfix_FullDataset_TrainDSstudy_pt_90.0",
    100: "Delphes_020425_test_PU_PFfix_part0/batch_eval_2k/DelphesPFfix_FullDataset_TrainDSstudy",
}
'''
print("PLOTTING QCD")
inputs = {
    30: "QCD_test_part0/batch_eval_2k/DelphesPFfix_FullDataset_TrainDSstudy_QCD_pt_30.0",
    40: "QCD_test_part0/batch_eval_2k/DelphesPFfix_FullDataset_TrainDSstudy_QCD_pt_40.0",
    50: "QCD_test_part0/batch_eval_2k/DelphesPFfix_FullDataset_TrainDSstudy_QCD_pt_50.0",
    60: "QCD_test_part0/batch_eval_2k/DelphesPFfix_FullDataset_TrainDSstudy_QCD_pt_60.0",
    70: "QCD_test_part0/batch_eval_2k/DelphesPFfix_FullDataset_TrainDSstudy_QCD_pt_70.0",
    80: "QCD_test_part0/batch_eval_2k/DelphesPFfix_FullDataset_TrainDSstudy_QCD_pt_80.0",
    90: "QCD_test_part0/batch_eval_2k/DelphesPFfix_FullDataset_TrainDSstudy_QCD_pt_90.0",
    100: "QCD_test_part0/batch_eval_2k/DelphesPFfix_FullDataset_TrainDSstudy_QCD"
}
'''
files = {
    key: pickle.load(open(os.path.join(get_path(value, "results"), "precision_recall.pkl"), "rb")) for key, value in inputs.items()
}

titles = {key: set(value.keys()) for key, value in files.items()}
# make a set of the intersections of titles
intersections = sorted(list(set.intersection(*titles.values())))


titles_to_plot = {
    "AK, R=0.8": ["AK8", "gray"],
    "GT_R=0.8 LGATr_GP_IRC_S_50k_s12900, sc. (aug)": ["LGATr_GP_IRC_S", "red"],
    "GT_R=0.8 LGATr_GP_50k_s25020, sc. (aug)": ["LGATr_GP", "purple"],
    "GT_R=0.8 base_LGATr_s50000, sc.": ["LGATr", "orange"]
} # To plot different variations of the model

print("QCD") # colors=   [{"base_LGATr": "orange", "LGATr_700_07": "red", "LGATr_QCD": "purple", "LGATr_700_07+900_03": "blue", "LGATr_700_07+900_03+QCD": "green", "AK8": "gray"}, {"base_LGATr": "LGATr_900_03"}],
titles_to_plot = {
    "AK, R=0.8": ["AK8", "gray"],

    "GT_R=0.8 base_LGATr_s50000, sc.": ["LGATr_900_03", "orange"],
    "GT_R=0.8 LGATr_QCD_s50000, sc.": ["LGATr_QCD", "purple"],
    "GT_R=0.8 LGATr_700_07_s50000, sc.": ["LGATr_700_07", "red"],
    "GT_R=0.8 LGATr_700_07+900_03_s50000, sc.": ["LGATr_700_07+900_03", "blue"],
    "GT_R=0.8 LGATr_700_07+900_03+QCD_s50000, sc.": ["LGATr_700_07+900_03+QCD", "green"],
}

titles_to_plot = {
    "AK, R=0.8": ["AK8", "gray"],
    "GT_R=0.8 LGATr_GP_IRC_S_50k_s12900, sc. (aug)": ["LGATr_900_03", "orange"],
    "GT_R=0.8 LGATr_GP_IRC_S_QCD_s24000, sc. (aug)": ["LGATr_QCD", "purple"],
    "GT_R=0.8 LGATr_GP_IRC_S_700_07_s24000, sc. (aug)": ["LGATr_700_07", "red"],
    "GT_R=0.8 LGATr_GP_IRC_S_700_07+900_03_s24000, sc. (aug)": ["LGATr_700_07+900_03", "blue"],
    "GT_R=0.8 LGATr_GP_IRC_S_700_07+900_03+QCD_s24000, sc. (aug)": ["LGATr_700_07+900_03+QCD", "green"],
}


intersections = sorted(list(titles_to_plot.keys()))

output_dirs = []
for _, value in inputs.items():
    output_dirs.append(get_path(value, "results"))
result = files[100][intersections[0]]
mediator_masses = sorted(list(result.keys()))

r_invs = sorted(list(set([rinv for mMed in result for mDark in result[mMed] for rinv in result[mMed][mDark]])))
sz = 4

#fig, ax = plt.subplots(len(inputs), len(titles_to_plot), figsize=(sz*len(titles_to_plot), sz*len(inputs)))
fig, ax = plt.subplots(len(mediator_masses), len(r_invs), figsize=(sz*len(r_invs), sz*len(mediator_masses)))
figp, axp = plt.subplots(len(mediator_masses), len(r_invs), figsize=(sz*len(r_invs), sz*len(mediator_masses)))
figr, axr = plt.subplots(len(mediator_masses), len(r_invs), figsize=(sz*len(r_invs), sz*len(mediator_masses)))

if len(r_invs) == 1 and len(mediator_masses) == 1:
    ax = np.array([[ax]])
    axp = np.array([[axp]])
    axr = np.array([[axr]])
grids = set()

for i, mMed in enumerate(mediator_masses):
    for j, rInv in enumerate(r_invs):
        for k, title in enumerate(intersections):
            label, color = titles_to_plot[title]
            pts = sorted(list(inputs.keys()))
            precisions = []
            recalls = []
            f1_scores = []
            for pt in pts:
                precision, recall = files[pt][title][mMed][20][rInv]
                precisions.append(precision)
                recalls.append(recall)
                f1_score = 2 * precision * recall / (precision + recall)
                f1_scores.append(f1_score)
            ax[i, j].plot(pts, f1_scores, ".-", label=label, color=color)
            axp[i, j].plot(pts, precisions, ".-", label=label, color=color)
            axr[i, j].plot(pts, recalls, ".-", label=label, color=color)
            ax[i, j].set_title(f"$m_{{Z'}} = {mMed}$ GeV, $r_{{inv.}}$ = {rInv}")
            ax[i, j].set_xlabel("$p_T^{cutoff}$")
            axp[i, j].set_title(f"$m_{{Z'}} = {mMed}$ GeV, $r_{{inv.}}$ = {rInv}")
            axp[i, j].set_xlabel("$p_T^{cutoff}$")
            axr[i, j].set_title(f"$m_{{Z'}} = {mMed}$ GeV, $r_{{inv.}}$ = {rInv}")
            axr[i, j].set_xlabel("$p_T^{cutoff}$")
            ax[i, j].set_ylabel("$F_1$ score")
            axp[i, j].set_ylabel("Precision")
            axr[i, j].set_ylabel("Recall")

            ax[i, j].legend()
            axp[i, j].legend()
            axr[i, j].legend()
            if (i, j) not in grids:
                ax[i, j].grid()
                axp[i, j].grid()
                axr[i, j].grid()
            grids.add((i, j))


for f in output_dirs:
    fig.tight_layout()
    fname = os.path.join(f, "pt_cutoff_vs_f1_score.pdf")
    fig.tight_layout()
    fig.savefig(fname)
    print("saved to", fname)

    fname = os.path.join(f, "pt_cutoff_vs_precision.pdf")
    figp.tight_layout()
    figp.savefig(fname)
    fname = os.path.join(f, "pt_cutoff_vs_recall.pdf")
    figr.tight_layout()
    figr.savefig(fname)

