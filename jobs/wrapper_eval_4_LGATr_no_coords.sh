#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpu               # Specify the partition
#SBATCH --account=gpu_gres            # Specify the account
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=40:00:00
#SBATCH --job-name=SVJFT  # Name the job
#SBATCH --gres=gpu:1
#SBATCH --output=jobs/tr4_$1.log
#SBATCH --error=jobs/tr4_$1.log
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi

srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m src.train -test scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-700_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-900_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1500_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-800_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1000_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1000_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-3000_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-800_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-700_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-900_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-3000_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-800_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1500_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1000_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1500_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-3000_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-900_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-700_mDark-20_rinv-0.7    -net src/models/LGATr/lgatr.py -bs 64 --gpus 0 --run-name Eval_LGATr_no_coords_loss_1 --val-dataset-size 100 --num-epochs 1000 --attr-loss-weight 0.1 --coord-loss-weight 0 --beta-type pt+bc --spatial-part-only --num-workers 0 --load-model-weights train/lgatr_no_coord_loss_1_2025_01_16_11_42_06/step_40000_epoch_30.ckpt

exit 0
EOT


# Run: bash jobs/wrapper_eval_4_LGATr_no_coords.sh <job_name>
