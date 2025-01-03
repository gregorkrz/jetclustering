#!/bin/bash
#SBATCH --partition=gpu           # Specify the partition
#SBATCH --account=gpu_gres                  # Specify the account
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=10:00:00               # Set the time limit to 1 hour
#SBATCH --job-name=SVJtest  # Name the job
#SBATCH --output=jobs/eval_1_output.log       # Redirect stdout to a log file
#SBATCH --error=jobs/eval_1_error.log         # Redirect stderr to a log file
#SBATCH --gres=gpu:1
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://dologarcia/gatr:v0 python -m src.train -test scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-900_mDark-20_rinv-0.3 -net src/models/GATr/Gatr.py -bs 64 --gpus 0 --run-name Test_betaPt_BC --beta-type pt+bc --load-model-weights train/OC_betaPt_BC_train_2024_12_27_18_00_40/step_10000_epoch_8.ckpt --predict

# Run the job:
# sbatch jobs/eval_1.slurm
