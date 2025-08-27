import pickle
import os
from src.utils.paths import get_path
from src.utils.utils import CPU_Unpickler
import argparse
from src.jetfinder.clustering import get_clustering_labels, get_clustering_labels_dbscan
import torch
# keeping the clustering script here for now, so that it's separated from the GPU-heavy tasks like inference (clustering may be changed frequently...)

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True) # train/Eval_eval_19March2025_small_aug_vanishing_momentum_Qcap05_p1e-2_reprod_1_2025_03_30_16_20_37_779
# train/Eval_eval_19March2025_small_aug_vanishing_momentum_Qcap05_p1e-2_reprod_1_2025_03_30_16_20_39_153
# train/Eval_eval_19March2025_pt1e-2_500particles_2025_04_01_11_57_07_994

# python -m scripts.compute_clustering --input  train/Eval_eval_19March2025_pt1e-2_500particles_2025_04_01_11_57_07_994 --output-suffix DefaultParams --min-cluster-size 15 --min-samples 5 --epsilon 0.1 --overwrite

# python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_FT_PL_2025_04_01_18_23_46_933 --output-suffix FT --min-cluster-size 15 --min-samples 1 --epsilon 0.3
# python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_FT_PL_2025_04_01_18_23_53_208 --output-suffix FT --min-cluster-size 15 --min-samples 1 --epsilon 0.3


# python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_FT_PL_2025_04_02_12_31_39_996 --output-suffix FT --min-cluster-size 15 --min-samples 1 --epsilon 0.3
# python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_FT_PL_2025_04_02_12_53_44_489 --output-suffix FT --min-cluster-size 15 --min-samples 1 --epsilon 0.3
# python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_FT_PL_2025_04_02_13_13_02_174 --output-suffix FT --min-cluster-size 15 --min-samples 1 --epsilon 0.3
# python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_FT_PL_2025_04_02_13_02_00_799 --output-suffix FT --min-cluster-size 15 --min-samples 1 --epsilon 0.3



# python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_FT_PL_2025_04_02_14_40_58_35 --output-suffix FT --min-cluster-size 15 --min-samples 1 --epsilon 0.3
# python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_FT_PL_2025_04_02_14_47_23_671 --output-suffix FT --min-cluster-size 15 --min-samples 1 --epsilon 0.3
# python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_FT_PL_2025_04_02_14_51_32_144 --output-suffix FT --min-cluster-size 15 --min-samples 1 --epsilon 0.3
# python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_FT_PL_2025_04_02_14_28_33_421 --output-suffix FT --min-cluster-size 15 --min-samples 1 --epsilon 0.3

# python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_FT_PL_2025_04_02_21_22_21_86 --output-suffix FT --min-cluster-size 15 --min-samples 1 --epsilon 0.3
# python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_FT_PL_2025_04_02_21_22_24_133 --output-suffix FT --min-cluster-size 15 --min-samples 1 --epsilon 0.3


#
parser.add_argument("--output-suffix", type=str, required=False, default="MinSamples0")
parser.add_argument("--min-cluster-size", type=int, default=2)
parser.add_argument("--min-samples", type=int, default=1)
parser.add_argument("--epsilon", type=float, default=0.3)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--spatial-part-only", action="store_true")
parser.add_argument("--dbscan", action="store_true", help="Use DBSCAN (with pt weights) instead of HDBSCAN. Only epsilon and min-samples would then be considered for clustering.")
parser.add_argument("--pt-hdbscan", action="store_true", help="Use the special distance function in HDBSCAN that is distance * min(pt1, pt2)")

args = parser.parse_args()
path = get_path(args.input, "results", fallback=True)

# For DBSCAN tests
"""
python -m scripts.compute_clustering --output-suffix dbscan_pt --min-cluster-size 4 --epsilon 0.1 --spatial-part-only --dbscan --input train/1
"""

for file in os.listdir(path):
    if file.startswith("eval_") and file.endswith(".pkl"):
        print("Computing clusters for file", file)
        result = CPU_Unpickler(open(os.path.join(path, file), "rb")).load()
        file_number = file.split("_")[1].split(".")[0]
        labels_path = os.path.join(path, "clustering_{}_{}.pkl".format(args.output_suffix, file_number))
        if not os.path.exists(labels_path) or args.overwrite:
            #dataset = EventDataset.from_directory(result["filename"], mmap=True)
            if result["pred"].shape[1] == 4:
                coords = result["pred"][:, :3]
            else:
                coords = result["pred"][:, :4]
                if args.spatial_part_only:
                    coords = coords[:, 1:4]
            if args.dbscan:
                labels = get_clustering_labels_dbscan(coords, result["pt"], result["event_idx"],
                                                      min_samples=args.min_samples, epsilon=args.epsilon)
            else:
                pt = None
                if args.pt_hdbscan:
                    pt = result["pt"]
                labels = torch.tensor(get_clustering_labels(coords, result["event_idx"], min_cluster_size=args.min_cluster_size,
                                               min_samples=args.min_samples, epsilon=args.epsilon, pt=pt, bar=True))
            with open(labels_path, "wb") as f:
                pickle.dump(labels, f)
            print("Saved labels to", labels_path)
        else:
            print("Labels already exist for this file at", labels_path)
