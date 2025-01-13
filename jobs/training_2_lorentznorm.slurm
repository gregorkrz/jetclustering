#!/bin/bash
#SBATCH --partition=gpu               # Specify the partition
#SBATCH --account=gpu_gres            # Specify the account
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=15:00:00               # Set the time limit to 1 hour
#SBATCH --job-name=SVJtr2  # Name the job
#SBATCH --output=jobs/training_2L_output1.log       # Redirect stdout to a log file
#SBATCH --error=jobs/training_2L_error11.log         # Redirect stderr to a log file
#SBATCH --gres=gpu:1
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m src.train -train scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-900_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 -val scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-900_mDark-20_rinv-0.3 -net src/models/LGATr/lgatr.py -bs 64 --gpus 0 --run-name Train_LGATr_lorentznorm --val-dataset-size 100 --num-epochs 1000  --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --lorentz-norm

# Overfit test

singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m src.train -train scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-900_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 -val scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-900_mDark-20_rinv-0.3 -net src/models/LGATr/lgatr.py -bs 16 --gpus 0 --run-name OverfitTestTrain_LGATr_lorentznorm_from_ckpt --val-dataset-size 16 --train-dataset-size 16 --num-epochs 15000  --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --lorentz-norm --load-model-weights train/OverfitTestTrain_LGATr_lorentznorm_2025_01_06_20_00_58/step_9000_epoch_9000.ckpt

# Eval overfit
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m src.train -train scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-900_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 -val scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-900_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 -net src/models/LGATr/lgatr.py -bs 16 --gpus 0 --run-name OverfitTest_LGATr --val-dataset-size 16  --train-dataset-size 16 --num-epochs 1  --attr-loss-weight 0.01 --coord-loss-weight 0.1 --beta-type pt+bc --load-model-weights  train/OverfitTest_LGATr_2025_01_06_12_58_07/step_8000_epoch_8000.ckpt
# Run the job:
# sbatch jobs/training_2_lorentznorm.slurm
