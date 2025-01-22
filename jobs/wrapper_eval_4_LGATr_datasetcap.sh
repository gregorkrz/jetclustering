#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpu               # Specify the partition
#SBATCH --account=gpu_gres            # Specify the account
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=05:00:00
#SBATCH --job-name=SVJFT  # Name the job
#SBATCH --gres=gpu:1
#SBATCH --output=jobs/eval_Tr_dsc4c_$1_out.log
#SBATCH --error=jobs/Eval_tr_dsc_$1_err.log
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi

srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m src.train --num-workers 0 -test scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-700_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-900_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1500_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-800_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1000_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1000_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-3000_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-800_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-700_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-900_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-3000_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-800_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1500_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1000_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1500_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-3000_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-900_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-700_mDark-20_rinv-0.7   -net src/models/LGATr/lgatr.py -bs 64 --gpus 0 --run-name lgatr_eval_ds_cap_$1  --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --spatial-part-only --load-model-weights $2

exit
EOT

# args: dscap, ckpt
#bash jobs/wrapper_eval_4_LGATr_datasetcap.sh 1000_S15k  train/lgatr_ds_cap_1000_2025_01_20_21_36_05/step_15000_epoch_1000.ckpt
#bash jobs/wrapper_eval_4_LGATr_datasetcap.sh 5000_S15k train/lgatr_ds_cap_5000_2025_01_20_21_37_19/step_15000_epoch_193.ckpt


# for the 40k steps:
# for 1000 dscap: step 22k of lgatr_CONT_ds_cap_1000_2025_01_21_19_41_51
# for 5000 dscap: step 20k of lgatr_CONT_ds_cap_5000_2025_01_21_19_46_13

# train/lgatr_CONT_ds_cap_5000_2025_01_21_19_46_13/step_20000_epoch_257.ckpt
#bash jobs/wrapper_eval_4_LGATr_datasetcap.sh 5000_S40k train/lgatr_CONT_ds_cap_5000_2025_01_21_19_46_13/step_20000_epoch_257.ckpt

