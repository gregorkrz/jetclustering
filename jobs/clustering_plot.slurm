#!/bin/bash
#SBATCH --partition=standard           # Specify the partition
#SBATCH --account=t3                  # Specify the account
#SBATCH --mem=10000                   # Request 10GB of memory
#SBATCH --time=00:30:00               # Set the time limit to 1 hour
#SBATCH --job-name=SVJ_clustering  # Name the job
#SBATCH --output=jobs/clustering_out1.log       # Redirect stdout to a log file
#SBATCH --error=jobs/clustering_err1.log         # Redirect stderr to a log file
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --input train/Test_betaPt_BC_all_datasets_2025_01_07_17_50_45


# Eval of the model trained on m=900 rinv=0.7
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.plot_multiple_models_clustering

# Run the job:
# sbatch jobs/clustering_plot.slurm
