#!/bin/bash
#SBATCH --partition=standard           # Specify the partition
#SBATCH --account=t3                  # Specify the account
#SBATCH --mem=15000                   # Request 10GB of memory
#SBATCH --time=01:30:00               # Set the time limit to 1 hour
#SBATCH --job-name=preprocess_svj_dataset  # Name the job
#SBATCH --output=jobs/preprocess_v0_output.log       # Redirect stdout to a log file
#SBATCH --error=jobs/preprocess_v0_error.log         # Redirect stderr to a log file

# Load the Singularity environment

source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
#singularity  shell  -B /t3home/gkrzmanc/  -B /work/gkrzmanc/ --nv docker://dologarcia/gatr:v0
# Run the Python script
#python -m src.preprocessing.preprocess_dataset --input SVJ_std_UL2018_scouting_test_large --output SVJ_std_UL2018_scouting_test_large
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://dologarcia/gatr:v0 python -m src.preprocessing.preprocess_dataset --input scouting_PFNano_signals/SVJ_hadronic_std --output scouting_PFNano_signals/SVJ_hadronic_std
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://dologarcia/gatr:v0 python -m src.preprocessing.preprocess_dataset --input scouting_PFNano_signals/SVJ_hadronic_std2 --output scouting_PFNano_signals/SVJ_hadronic_std2
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://dologarcia/gatr:v0 python -m src.preprocessing.preprocess_dataset --input scouting_PFNano_signals/SVJ_hadronic_std3 --output scouting_PFNano_signals/SVJ_hadronic_std3


# Run the job: sbatch jobs/preprocess_v0.slurm
