#!/bin/bash
#SBATCH --partition=standard           # Specify the partition
#SBATCH --account=t3                  # Specify the account
#SBATCH --mem=5000                   # Request 10GB of memory
#SBATCH --time=01:00:00               # Set the time limit to 1 hour
#SBATCH --job-name=FT_clustering  # Name the job
#SBATCH --output=jobs/tune_clustering_lgatr_output2.log       # Redirect stdout to a log file
#SBATCH --error=jobs/tune_clustering_lgatr2.log         # Redirect stderr to a log file
#SBATCH -n 1
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
# CosSim
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.tune_clustering --input "/work/gkrzmanc/jetclustering/results/train/Eval_IRCLossDebug_Reproduce_GluonFix_2025_04_14_10_24_23_167" --dataset-cap 2000 --spatial-components-only


# sbatch jobs/tune_clustering_LGATr_March.slurm



