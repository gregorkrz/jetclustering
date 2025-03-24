#!/bin/bash
#SBATCH --partition=long           # Specify the partition
#SBATCH --account=t3                  # Specify the account
#SBATCH --mem=60000                   # Request 60GB of memory
#SBATCH --time=24:00:00               # Set the time limit to 1 hour
#SBATCH --job-name=preprocess_svj_dataset  # Name the job
#SBATCH --output=jobs/preprocess_v2_output.log       # Redirect stdout to a log file
#SBATCH --error=jobs/preprocess_v2_error.log         # Redirect stderr to a log file

# Load the Singularity environment

source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache

# Run the Python script
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ -B /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc --nv docker://gkrz/lgatr:v3 python -m src.preprocessing.preprocess_dataset --input Feb26_2025_E1000_N500_folders --output Feb26_2025_E1000_N500_full --v2 --overwrite

# Run the job: sbatch jobs/preprocess_v2.slurm
