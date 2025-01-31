#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpu               # Specify the partition
#SBATCH --account=gpu_gres            # Specify the account
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=40:00:00
#SBATCH --job-name=SVJFT  # Name the job
#SBATCH --gres=gpu:1
#SBATCH --output=jobs/tr4b_$1.log
#SBATCH --error=jobs/tr4b_$1.log
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi
export WANDB_DIR=/scratch/slurm/$SLURM_JOB_ID

srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m src.train -train scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-900_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 -val scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-900_mDark-20_rinv-0.3 -net src/models/LGATr/lgatr.py -bs 64 --gpus 0 --run-name lgatr_CONT_ds_cap_$1 --val-dataset-size 1000 --train-dataset-size $1 --num-epochs 99999 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --spatial-part-only --gt-radius 1.4 --num-workers 0 --load-model-weights train/lgatr_CONT_ds_cap_5000_2025_01_21_19_46_13/step_21000_epoch_270.ckpt

exit 0
EOT


# bash jobs/wrapper_training_4_LGATr_datasetcap.sh 1000
# bash jobs/wrapper_training_4_LGATr_datasetcap.sh 5000

# Still need to run:
# bash jobs/wrapper_training_4_LGATr_datasetcap.sh 500
# bash jobs/wrapper_training_4_LGATr_datasetcap.sh 10000
# # train/lgatr_CONT_ds_cap_5000_2025_01_21_19_46_13/step_21000_epoch_270.ckpt