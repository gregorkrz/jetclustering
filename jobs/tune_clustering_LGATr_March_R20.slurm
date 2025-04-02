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
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.tune_clustering --input "/work/gkrzmanc/jetclustering/results/train/Eval_eval_19March2025_pt1e-2_500particles_2025_04_01_14_04_05_945" --dataset-cap 1500 --spatial-components-only


# sbatch jobs/tune_clustering_LGATr_March_R20.slurm



