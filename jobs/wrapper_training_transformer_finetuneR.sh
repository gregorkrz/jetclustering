#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpu               # Specify the partition
#SBATCH --account=gpu_gres            # Specify the account
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=40:00:00
#SBATCH --job-name=SVJFT  # Name the job
#SBATCH --gres=gpu:1
#SBATCH --output=jobs/tr4T_$1.log
#SBATCH --error=jobs/tr4T_$1.log
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi

srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m src.train -train scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-900_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 -val scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-900_mDark-20_rinv-0.3 -net src/models/transformer/transformer.py -bs 64 --gpus 0 --run-name FT_GT_R_Transformer --val-dataset-size 1024 --num-steps 40000 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --gt-radius $1

exit 0
EOT

# bash jobs/wrapper_training_transformer_finetuneR.sh 1.4
# bash jobs/wrapper_training_transformer_finetuneR.sh 2.0
# bash jobs/wrapper_training_transformer_finetuneR.sh 0.7
# bash jobs/wrapper_training_transformer_finetuneR.sh 0.8
# bash jobs/wrapper_training_transformer_finetuneR.sh 0.9



#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m src.train -train scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-900_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 -val scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-900_mDark-20_rinv-0.3 -net src/models/LGATr/lgatr.py -bs 64 --gpus 0 --run-name FT_GT_R_LGATr_CONT --val-dataset-size 100 --num-epochs 1000 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --spatial-part-only --gt-radius $1 --load-model-weights train/FT_GT_R_LGATr_2025_01_20_12_00_14/step_30000_epoch_23.ckpt
# --load-model-weights train/FT_GT_R_LGATr_2025_01_20_12_00_16/step_30000_epoch_23.ckpt