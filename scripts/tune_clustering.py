import pickle
import os
from src.utils.paths import get_path
from src.utils.utils import CPU_Unpickler
import argparse
from src.jetfinder.clustering import get_clustering_labels
import optuna
from src.dataset.dataset import EventDataset
from src.evaluation.clustering_metrics import compute_f1_score
import torch

import warnings
warnings.filterwarnings("ignore")

# filename = get_path("/work/USER/jetclustering/results/train/Test_betaPt_BC_2025_01_03_15_07_14/eval_0.pkl", "results")
# for rinv=0.7, see /work/USER/jetclustering/results/train/Test_betaPt_BC_rinv07_2025_01_03_15_38_58
# keeping the clustering script here for now, so that it's separated from the GPU-heavy tasks like inference (clustering may be changed frequently...)
# parameters: min-cluster-size: [5, 30]
#             min-samples: [2, 30]
#             epsilon: [0.01, 0.5]

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--dataset", type=int, required=False, default=11) # Which dataset to optimize on
parser.add_argument("--dataset-cap", type=int, required=False, default=-1)
parser.add_argument("--spatial-components-only", "-spatial-only", action="store_true")
parser.add_argument("--lorentz-cos-sim", action="store_true")
parser.add_argument("--cos-sim", action="store_true")
parser.add_argument("--normalize", action="store_true")

# --input train/  --dataset-cap 1000 --spatial-components-only
args = parser.parse_args()
path = get_path(args.input, "results")
suffix = ""
if args.spatial_components_only:
    suffix = "_sp_comp_only"
#if args.lorentz_norm:
#    suffix = "_lorentz_norm"
if args.lorentz_cos_sim:
    suffix = "_lorentz_cos_sim"
if args.cos_sim:
    suffix = "_cos_sim"
if args.normalize:
    suffix = "_norm"

study_file = os.path.join(path, "clustering_tuning_{}{}.log".format(args.dataset, suffix))

study_exists = os.path.exists(study_file)
storage = optuna.storages.JournalStorage(
    optuna.storages.journal.JournalFileBackend(study_file)
)

if study_exists:
    study = optuna.load_study(storage=storage, study_name="clustering")
else:
    study = optuna.create_study(storage=storage, study_name="clustering", direction="maximize")

eval_result_file = os.path.join(path, "eval_{}.pkl".format(args.dataset))
eval_result = CPU_Unpickler(open(eval_result_file, "rb")).load()

dataset_cap = args.dataset_cap


def objective(trial):
    min_clust_size = trial.suggest_int("min_cluster_size", 2, 20)
    min_samples = trial.suggest_int("min_samples", 0, 10)
    epsilon = trial.suggest_uniform("epsilon", 0.01, 0.5)
    print("Starting trial with parameters:", trial.params)
    suffix = "{}-{}-{}".format(min_clust_size, min_samples, epsilon)
    if args.spatial_components_only:
        suffix = "sp-" + suffix
    #if args.lorentz_norm:
    #    suffix = "ln-" + suffix
    if args.cos_sim:
        suffix = "cs-" + suffix
    if args.lorentz_cos_sim:
        suffix = "lcs-" + suffix
    if args.normalize:
        suffix = "norm-" + suffix
    clustering_file = os.path.join(path, "clustering_{}_{}.pkl".format(suffix, args.dataset))
    if not os.path.exists(clustering_file):
        if eval_result["pred"].shape[1] == 4:
            coords = eval_result["pred"][:, :3]
        else:
            if args.spatial_components_only or args.cos_sim:
                coords = eval_result["pred"][:, 1:4]
            else:
                coords = eval_result["pred"][:, :4]
        event_idx = eval_result["event_idx"]
        if dataset_cap > 0:
            filt = event_idx < dataset_cap
            event_idx = event_idx[filt]
            coords = coords[filt]
        if args.cos_sim or args.normalize:
            coords = coords / torch.norm(coords, dim=1, keepdim=True)
        labels = get_clustering_labels(coords, event_idx, min_cluster_size=min_clust_size,
                                       min_samples=min_samples, epsilon=epsilon, bar=True,
                                       lorentz_cos_sim=args.lorentz_cos_sim,
                                       cos_sim=args.cos_sim)
        with open(clustering_file, "wb") as f:
            pickle.dump(labels, f)
        print("Clustering saved to", clustering_file)
    #else:
    #    labels = pickle.load(open(clustering_file, "rb"))
    print("Dataset:", eval_result["filename"])
    dataset = EventDataset.from_directory(eval_result["filename"],
                                          model_clusters_file=clustering_file,
                                          model_output_file=eval_result_file,
                                          include_model_jets_unfiltered=True, parton_level=True, aug_soft=True)
    score = compute_f1_score(dataset, dataset_cap=dataset_cap)
    print("F1 score for", suffix, ":", score)
    return score

study.optimize(objective, n_trials=100)
print(f"Best params is {study.best_params} with value {study.best_value}")

