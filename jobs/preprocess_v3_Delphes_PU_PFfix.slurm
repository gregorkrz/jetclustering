#!/bin/bash
#SBATCH --partition=standard           # Specify the partition
#SBATCH --account=t3                  # Specify the account
#SBATCH --mem=60000                   # Request 60GB of memory
#SBATCH --time=06:00:00               # Set the time limit to 1 hour
#SBATCH --job-name=preprocess_svj_dataset  # Name the job
#SBATCH --output=jobs/preprocess_v3_Delphes_output.log       # Redirect stdout to a log file
#SBATCH --error=jobs/preprocess_v3_Delphes_error.log         # Redirect stderr to a log file
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gkrzmanc@student.ethz.ch
# Load the Singularity environment

source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache

# Run the Python script

singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ -B /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc --nv docker://gkrz/lgatr:v3 python -m src.preprocessing.preprocess_dataset --input Delphes_020425_test_PU_PFfix --output Delphes_020425_test_PU_PFfix --v2 --overwrite --delphes

# Run the job: sbatch jobs/preprocess_v3_Delphes_PU_PFfix.slurm


