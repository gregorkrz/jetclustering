#!/bin/bash
#SBATCH --partition=short           # Specify the partition
#SBATCH --account=t3                  # Specify the account
#SBATCH --mem=10000                   # Request 10GB of memory
#SBATCH --time=01:00:00               # Set the time limit to 1 hour
#SBATCH --job-name=SVJ_clustering  # Name the job
#SBATCH --output=jobs/clustering_out.log       # Redirect stdout to a log file
#SBATCH --error=jobs/clustering_err.log         # Redirect stderr to a log file
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache

srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.plot_multiple_models_clustering

# Run the job:
# sbatch jobs/clustering_503.slurm
