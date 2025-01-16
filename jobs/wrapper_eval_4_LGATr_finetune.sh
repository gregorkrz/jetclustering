#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpu               # Specify the partition
#SBATCH --account=gpu_gres            # Specify the account
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=05:00:00
#SBATCH --job-name=SVJFT  # Name the job
#SBATCH --gres=gpu:1
#SBATCH --output=jobs/eval4_$1_out.log
#SBATCH --error=jobs/eval4_$1.log
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi

srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m src.train --num-workers 0 -test scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-700_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-900_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1500_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-800_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1000_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1000_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-3000_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-800_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-700_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-900_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-3000_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-800_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1500_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1000_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1500_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-3000_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-900_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-700_mDark-20_rinv-0.7   -net src/models/LGATr/lgatr.py -bs 64 --gpus 0 --run-name Eval_GT_R_lgatr_$1 --num-epochs 1000 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --spatial-part-only --load-model-weights train/$2/step_40000_epoch_30.ckpt

exit
EOT


# run: bash jobs/wrapper_eval_4_GATr_finetune.sh R07 run_name

# Runs to evaluate:
#R10 FT_GT_R_LGATr_2025_01_14_15_41_06
#R06 FT_GT_R_LGATr_2025_01_14_15_40_59
#R09 FT_GT_R_LGATr_2025_01_14_15_40_56
#R07 FT_GT_R_LGATr_2025_01_14_14_58_18

# Commands:
# bash jobs/wrapper_eval_4_LGATr_finetune.sh R10 FT_GT_R_LGATr_2025_01_14_15_41_06
# bash jobs/wrapper_eval_4_LGATr_finetune.sh R06 FT_GT_R_LGATr_2025_01_14_15_40_59
# bash jobs/wrapper_eval_4_LGATr_finetune.sh R09 FT_GT_R_LGATr_2025_01_14_15_40_56
# bash jobs/wrapper_eval_4_LGATr_finetune.sh R07 FT_GT_R_LGATr_2025_01_14_14_58_18




# Big training eval
# train/Train_LGATr_SB_All_data_CONT_2025_01_15_10_38_20/step_21000_epoch_3.ckpt  # total: step 26k


