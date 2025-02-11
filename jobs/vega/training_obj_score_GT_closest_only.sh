#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpu               # Specify the partition
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=40:00:00
#SBATCH --job-name=SVJTransformerTraining  # Name the job
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres            # Specify the account
#SBATCH --output=jobs/vega/debug_training_transformer_out_$1_$2_$3.log
#SBATCH --error=jobs/vega/debug_training_transformer_err_$1_$2_$3.log

source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi

srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ -B /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc --nv docker://gkrz/lgatr:v3 python -m src.train -train scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-900_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 -val scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-900_mDark-20_rinv-0.3 -net src/models/LGATr/lgatr.py -bs 128  --gpus 0 --run-name scatter_add_Obj_Score_LGATr_$1_$2_$3 --val-dataset-size 1024 --num-steps 200000 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --gt-radius 0.8 --num-blocks $1 -mv-ch $2 -s-ch $3 --spatial-part-only  --train-objectness-score --load-model-weights train/LGATr_training_40k_8_16_64_2025_01_28_16_20_14/step_40000_epoch_30.ckpt --num-workers 0 --start-lr 1e-4 --lr-scheduler none -obj-score-gt closest_only

exit 0
EOT


# Args: n_blocks mv_channels s_channels (default: 10, 16, 64)
# bash jobs/vega/training_obj_score_GT_closest_only.sh  8 16 64

