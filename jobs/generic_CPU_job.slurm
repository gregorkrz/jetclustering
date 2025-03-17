#!/bin/bash
#SBATCH --partition=gpu           # Specify the partition
#SBATCH --mem=10000                   # Request 10GB of memory
#SBATCH --time=01:00:00               # Set the time limit to 1 hour
#SBATCH --job-name=SVJ  # Name the job
#SBATCH --output=jobs/cpu.log       # Redirect stdout to a log file
#SBATCH --error=jobs/cpuerr.log         # Redirect stderr to a log file
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache

singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --output-suffix hdbscan_4_05 --min-samples 2 --epsilon 0.5 --min-cluster-size 4 --spatial-part-only --input train/Eval_no_pid_eval_1_2025_03_05_14_35_14
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --output-suffix hdbscan_4_05 --min-samples 2 --epsilon 0.5 --min-cluster-size 4 --spatial-part-only --input train/Eval_no_pid_eval_1_2025_03_05_14_36_15
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --output-suffix hdbscan_4_05 --min-samples 2 --epsilon 0.5 --min-cluster-size 4 --spatial-part-only --input train/Eval_no_pid_eval_1_2025_03_05_14_37_53
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --output-suffix hdbscan_4_05 --min-samples 2 --epsilon 0.5 --min-cluster-size 4 --spatial-part-only --input train/Eval_no_pid_eval_1_2025_03_05_14_40_26
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --output-suffix hdbscan_4_05 --min-samples 2 --epsilon 0.5 --min-cluster-size 4 --spatial-part-only --input train/Eval_no_pid_eval_1_2025_03_05_14_40_27
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --output-suffix hdbscan_4_05 --min-samples 2 --epsilon 0.5 --min-cluster-size 4 --spatial-part-only --input train/Eval_no_pid_eval_1_2025_03_05_14_40_30
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --output-suffix hdbscan_4_05 --min-samples 2 --epsilon 0.5 --min-cluster-size 4 --spatial-part-only --input train/Eval_no_pid_eval_1_2025_03_05_14_40_31
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --output-suffix hdbscan_4_05 --min-samples 2 --epsilon 0.5 --min-cluster-size 4 --spatial-part-only --input train/Eval_no_pid_eval_1_2025_03_05_14_40_34
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --output-suffix hdbscan_4_05 --min-samples 2 --epsilon 0.5 --min-cluster-size 4 --spatial-part-only --input train/Eval_no_pid_eval_1_2025_03_05_14_41_16
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --output-suffix hdbscan_4_05 --min-samples 2 --epsilon 0.5 --min-cluster-size 4 --spatial-part-only --input train/Eval_no_pid_eval_1_2025_03_05_14_41_17
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --output-suffix hdbscan_4_05 --min-samples 2 --epsilon 0.5 --min-cluster-size 4 --spatial-part-only --input train/Eval_no_pid_eval_1_2025_03_05_14_41_37
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --output-suffix hdbscan_4_05 --min-samples 2 --epsilon 0.5 --min-cluster-size 4 --spatial-part-only --input train/Eval_no_pid_eval_1_2025_03_05_14_41_38

echo "Done"

