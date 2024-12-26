#!/bin/bash
#SBATCH --partition=gpu           # Specify the partition
#SBATCH --account=gpu_gres                  # Specify the account
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=03:00:00               # Set the time limit to 1 hour
#SBATCH --job-name=SVJ_training_01  # Name the job
#SBATCH --output=jobs/training_0_output1.log       # Redirect stdout to a log file
#SBATCH --error=jobs/training_0_error1.log         # Redirect stderr to a log file
#SBATCH --gres=gpu:1
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://dologarcia/gatr:v0 python -m src.train -train scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-900_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 -val scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-900_mDark-20_rinv-0.3 -net src/models/GATr/Gatr.py -bs 16 --gpus 0 --run-name Overfit_OC_qmin3_3Dclusters --train-dataset-size 16 --val-dataset-size 16 --num-epochs 10000 --attr-loss-weight 0.1 --load-model-weights train/Overfit_OC_qmin3_3Dclusters_2024_12_25_20_57_28/step_2000_epoch_2000.ckpt

# Run the job:
# sbatch jobs/training_0.slurm
