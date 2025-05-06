#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpu               # Specify the partition
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=48:00:00
#SBATCH --job-name=DelphesTrainSVJ  # Name the job
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres            # Specify the account
#SBATCH --output=jobs/vega/Dtr_lgatr_aug.out
#SBATCH --error=jobs/vega/Dtr_lgatr_aug.err

source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi

srun singularity exec  -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ -B /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc  --nv docker://gkrz/lgatr:v3 python -m src.train  -train Delphes_020425_train2_PU_PFfix_part0/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_PU_PFfix_part1/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_PU_PFfix_part2/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_PU_PFfix_part3/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_PU_PFfix_part4/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_PU_PFfix_part5/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_PU_PFfix_part6/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak  Delphes_020425_train2_PU_PFfix_part7/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak  Delphes_020425_train2_PU_PFfix_part8/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak  -val Delphes_020425_train2_PU_PFfix_part9/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak -val scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-700_mDark-20_rinv-0.7 -net src/models/LGATr/lgatr.py -bs 8 --gpus 0 --run-name LGATr_Aug --val-dataset-size 150 --num-steps 200000 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --gt-radius 0.8 --num-blocks 10 -mv-ch 16 -s-ch 64 --spatial-part-only --validation-steps 60 --no-pid --augment-soft-particles --load-model-weights train/LGATr_training_NoPID_Delphes_PU_PFfix_10_16_64_0.8_2025_05_03_18_35_53_134/step_70000_epoch_16.ckpt --num-workers 0

exit 0
EOT

# Args: n_blocks mv_channels s_channels radius (default: 10, 16, 64, 0.8)
# bash jobs/IRC_training/Delphes_training_t3_NoPID_augment.sh
