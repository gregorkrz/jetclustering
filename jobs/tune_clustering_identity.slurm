#!/bin/bash
#SBATCH --partition=standard           # Specify the partition
#SBATCH --account=t3                  # Specify the account
#SBATCH --mem=5000                   # Request 10GB of memory
#SBATCH --time=00:30:00             # Set the time limit to 1 hour
#SBATCH --job-name=FT_clustering  # Name the job
#SBATCH --output=jobs/tune_clustering_id_output.log       # Redirect stdout to a log file
#SBATCH --error=jobs/tune_clustering_id.log         # Redirect stderr to a log file
#SBATCH -n 1
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.tune_clustering --input "/work/gkrzmanc/jetclustering/results/train/Eval_Identity_ValSet_STD2_2025_01_13_17_59_01" --dataset-cap 10000

# sbatch jobs/tune_clustering_identity.slurm
