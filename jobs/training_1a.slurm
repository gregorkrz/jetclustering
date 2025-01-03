#!/bin/bash
#SBATCH --partition=gpu           # Specify the partition
#SBATCH --account=gpu_gres                  # Specify the account
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=10:00:00               # Set the time limit to 1 hour
#SBATCH --job-name=SVJtr1  # Name the job
#SBATCH --output=jobs/training_1a_output.log       # Redirect stdout to a log file
#SBATCH --error=jobs/training_1a_error.log         # Redirect stderr to a log file
#SBATCH --gres=gpu:1
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://dologarcia/gatr:v0 python -m src.train -train scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-900_mDark-20_rinv-0.5_alpha-peak_13TeV-pythia8_n-2000 -val scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-900_mDark-20_rinv-0.5 -net src/models/GATr/Gatr.py -bs 32 --gpus 0 --run-name OC_betaPt_BC_train --val-dataset-size 1000 --num-epochs 1000 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc

# Run the job:
# sbatch jobs/training_1a.slurm
