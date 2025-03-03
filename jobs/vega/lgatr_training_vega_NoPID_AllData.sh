#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpu               # Specify the partition
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=48:00:00
#SBATCH --job-name=SVJTransformerTraining  # Name the job
#SBATCH --gres=gpu:1
#SBATCH --account=s25t01-01-users             # Specify the account
#SBATCH --output=jobs/vega/TrLGATr_out_$1_$2_$3_R$4.log
#SBATCH --error=jobs/vega/TrLGATr_err_$1_$2_$3_R$4.log

source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi

srun singularity exec  -B /ceph/hpc/home/krzmancg --nv docker://gkrz/lgatr:v3 python -m src.train  -train scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-1100_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-1500_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-1100_mDark-20_rinv-0.5_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-1500_mDark-20_rinv-0.5_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-1100_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-1500_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-1200_mDark-20_rinv-0.5_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-700_mDark-20_rinv-0.5_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-1200_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-1300_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-800_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-1300_mDark-20_rinv-0.5_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-800_mDark-20_rinv-0.5_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-1300_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-800_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-1400_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-900_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-1400_mDark-20_rinv-0.5_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-900_mDark-20_rinv-0.5_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-1400_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-2000 scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-900_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-2000 -val scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-1000_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-1500_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-700_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-900_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-1000_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-1500_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-800_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-900_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-1000_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-700_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-800_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-900_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-1500_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-700_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-800_mDark-20_rinv-0.7   -net src/models/LGATr/lgatr.py -bs 64  --gpus 0 --run-name LGATr_training_NoPID_$1_$2_$3_$4_AllData --val-dataset-size 15000 --num-steps 200000 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --gt-radius $4 --num-blocks $1 -mv-ch $2 -s-ch $3 --spatial-part-only --validation-steps 2000 --no-pid

exit 0
EOT


# Args: n_blocks mv_channels s_channels radius (default: 10, 16, 64, 0.8)
# bash jobs/vega/lgatr_training_vega_NoPID_AllData.sh 10 16 64 0.8

