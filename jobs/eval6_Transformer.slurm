#!/bin/bash
#SBATCH --partition=gpu               # Specify the partition
#SBATCH --account=gpu_gres            # Specify the account
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=03:00:00
#SBATCH --job-name=SVJFT  # Name the job
#SBATCH --gres=gpu:1
#SBATCH --output=jobs/Eval6_output.log       # Redirect stdout to a log file
#SBATCH --error=jobs/Eval6_error.log         # Redirect stderr to a log file
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi

srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m src.train  -test scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-700_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-900_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1500_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-800_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1000_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1000_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-3000_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-800_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-700_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-900_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-3000_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-800_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1500_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1000_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1500_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-3000_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-900_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-700_mDark-20_rinv-0.7  -net src/models/transformer/transformer.py -bs 64 --gpus 0 --run-name Eval_Transformer --val-dataset-size 2000 --num-epochs 1000 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --gt-radius 0.8 --load-model-weights train/Train_Transformer_2025_01_16_15_15_11/step_170000_epoch_125.ckpt --num-workers 0
