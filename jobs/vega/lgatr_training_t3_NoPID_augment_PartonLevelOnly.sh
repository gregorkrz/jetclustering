#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpu               # Specify the partition
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=48:00:00
#SBATCH --job-name=SVJTraining  # Name the job
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres            # Specify the account
#SBATCH --output=jobs/vega/tr_lgatr_aug.out
#SBATCH --error=jobs/vega/tr_lgatr_aug.err

source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi

srun singularity exec  -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ -B /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc  --nv docker://gkrz/lgatr:v3 python -m src.train -train sFeb26_2025_E1000_N500_noPartonFilter_C_F/PFNano_s-channel_mMed-900_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000 -val Feb26_2025_E1000_N500_noPartonFilter_C_F/PFNano_s-channel_mMed-1000_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000  -net src/models/LGATr/lgatr.py -bs 16 --gpus 0 --run-name LGATr_training_NoPID_10_16_64_0.8_Aug_Finetune_PLOnly --val-dataset-size 15000 --num-steps 200000 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --gt-radius 0.8 --num-blocks 10 -mv-ch 16 -s-ch 64 --spatial-part-only --validation-steps 60 --no-pid -pl --load-model-weights train/LGATr_training_NoPID_10_16_64_0.8_2025_02_28_12_42_59/step_50000_epoch_37.ckpt --num-workers 0

exit 0
EOT

# Args: n_blocks mv_channels s_channels radius (default: 10, 16, 64, 0.8)
# bash jobs/vega/lgatr_training_t3_NoPID_augment_PartonLevelOnly.sh


