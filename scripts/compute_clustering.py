import pickle
import os
from src.utils.paths import get_path
from src.utils.utils import CPU_Unpickler
import argparse
from src.jetfinder.clustering import get_clustering_labels, get_clustering_labels_dbscan

# filename = get_path("/work/gkrzmanc/jetclustering/results/train/Test_betaPt_BC_2025_01_03_15_07_14/eval_0.pkl", "results")
# for rinv=0.7, see /work/gkrzmanc/jetclustering/results/train/Test_betaPt_BC_rinv07_2025_01_03_15_38_58
# keeping the clustering script here for now, so that it's separated from the GPU-heavy tasks like inference (clustering may be changed frequently...)

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True) # train/Eval_GT_R_lgatr_R14_2025_01_18_13_28_47
parser.add_argument("--output-suffix", type=str, required=False, default="") # SP
parser.add_argument("--min-cluster-size", type=int, default=11)
parser.add_argument("--min-samples", type=int, default=18)
parser.add_argument("--epsilon", type=float, default=0.48)
parser.add_argument("--spatial-part-only", action="store_true")
parser.add_argument("--dbscan", action="store_true", help="Use DBSCAN (with pt weights) instead of HDBSCAN. Only epsilon and min-samples would then be considered for clustering.")
parser.add_argument("--pt-hdbscan", action="store_true", help="Use the special distance function in HDBSCAN that is distance * min(pt1, pt2)")

args = parser.parse_args()
path = get_path(args.input, "results")

#dir_results = get_path("/work/gkrzmanc/jetclustering/results/train/Test_betaPt_BC_2025_01_03_15_07_14/eval_0.pkl", "results")
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
        if not os.path.exists(labels_path):
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
                labels = get_clustering_labels(coords, result["event_idx"], min_cluster_size=args.min_cluster_size,
                                               min_samples=args.min_samples, epsilon=args.epsilon, pt=pt, bar=True)
            with open(labels_path, "wb") as f:
                pickle.dump(labels, f)
            print("Saved labels to", labels_path)
        else:
            print("Labels already exist for this file")
