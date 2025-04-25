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
srun singularity exec  -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ -B /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc  --nv docker://gkrz/lgatr:v3 python -m src.train  -train Delphes_020425_train2_part0/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_part1/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_part2/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_part3/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_part4/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_part5/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_part6/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak  Delphes_020425_train2_part7/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak  Delphes_020425_train2_part8/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak  -val Delphes_020425_train2_part9/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak  -net src/models/LGATr/lgatr.py -bs 8 --gpus 0 --run-name CONT_Delphes_IRC_aug_SplitOnly --val-dataset-size 150 --num-steps 200000 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --gt-radius 0.8 --num-blocks 10 -mv-ch 16 -s-ch 64 --spatial-part-only --validation-steps 60 --no-pid --augment-soft-particles --load-model-weights train/Delphes_IRC_aug_SplitOnly_2025_04_20_15_50_33_553/step_14280_epoch_2.ckpt --num-workers 0 -irc


exit 0
EOT

# bash jobs/vega/Delphes_training_t3_NoPID_augment_IRC.sh
