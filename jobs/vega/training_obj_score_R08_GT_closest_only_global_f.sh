#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpu               # Specify the partition
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=40:00:00
#SBATCH --job-name=SVJTransformerTraining  # Name the job
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres            # Specify the account
#SBATCH --output=jobs/vega/debug_train_obj_score_R08.log
#SBATCH --error=jobs/vega/debug_train_err_obj_score_R08.log

source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi

srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ -B /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc --nv docker://gkrz/lgatr:v3 python -m src.train -train scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-900_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 -val scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-900_mDark-20_rinv-0.3 -net src/models/LGATr/lgatr.py -bs 64 --gpus 0 --run-name R08_LGATr_Obj_Score_EvtLevelFeatures --val-dataset-size 2000 --num-epochs 99999 --num-steps 100000  --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --spatial-part-only --gt-radius 0.8 --load-model-weights train/LGATr_training_40k_8_16_64_2025_01_28_16_20_14/step_40000_epoch_30.ckpt --lr-scheduler none --train-objectness-score --start-lr  1e-4 --num-workers 0 -obj-score-gt closest_only --global-features-obj-score --num-blocks 8 -mv-ch 16 -s-ch 64


exit 0
EOT
