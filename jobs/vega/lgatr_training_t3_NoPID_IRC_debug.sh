#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpu               # Specify the partition
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=48:00:00
#SBATCH --job-name=SVJTraining  # Name the job
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres            # Specify the account
#SBATCH --output=jobs/vega/tr_lgatr_IRC_debug.out
#SBATCH --error=jobs/vega/tr_lgatr_IRC_debug.err

source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
export CUDA_LAUNCH_BLOCKING=1
nvidia-smi

srun singularity exec  -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ -B /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc  --nv docker://gkrz/lgatr:v3 python -m src.train -train scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-900_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 -val scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-700_mDark-20_rinv-0.7 -net src/models/LGATr/lgatr.py -bs 8 --gpus 0 --run-name IRC_loss_Split_and_Noise_alternate_NoAug --val-dataset-size 10 --num-steps 200000 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --gt-radius 0.8 --num-blocks 10 -mv-ch 16 -s-ch 64 --spatial-part-only --validation-steps 60 --no-pid  --load-model-weights train/LGATr_training_NoPID_10_16_64_0.8_2025_02_28_12_42_59/step_50000_epoch_37.ckpt --num-workers 0 -irc

exit 0
EOT


# bash jobs/vega/lgatr_training_t3_NoPID_IRC_debug.sh

