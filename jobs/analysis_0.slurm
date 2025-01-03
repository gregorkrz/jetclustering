#!/bin/bash
#SBATCH --partition=standard           # Specify the partition
#SBATCH --account=t3                  # Specify the account
#SBATCH --mem=10000                   # Request 10GB of memory
#SBATCH --time=01:00:00               # Set the time limit to 1 hour
#SBATCH --job-name=SVJ_analysis_0  # Name the job
#SBATCH --output=jobs/analysis_0_output.log       # Redirect stdout to a log file
#SBATCH --error=jobs/analysis_0_error.log         # Redirect stderr to a log file
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://dologarcia/gatr:v0 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --dataset-cap 2000
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://dologarcia/gatr:v0 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/GenJets --dataset-cap 2000 --jets-object genjets
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://dologarcia/gatr:v0 python -m scripts.analysis.dataset_stats --input scouting_PFNano_signals2/SVJ_hadronic_std --dataset-cap 2000

# Run the job:
# sbatch jobs/analysis_0.slurm

