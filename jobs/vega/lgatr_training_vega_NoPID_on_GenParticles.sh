#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpu               # Specify the partition
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=48:00:00
#SBATCH --job-name=SVJTraining  # Name the job
#SBATCH --gres=gpu:1
#SBATCH --account=s25t01-01-users             # Specify the account
#SBATCH --output=jobs/vega/TrLGATr_GPout_$1_$2_$3_R$4.log
#SBATCH --error=jobs/vega/TrLGATr_GPerr_$1_$2_$3_R$4.log

source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi

srun singularity exec  -B /ceph/hpc/home/krzmancg --nv docker://gkrz/lgatr:v3 python -m src.train -train Feb26_2025_E1000_N500_full/PFNano_s-channel_mMed-900_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000  -val Feb26_2025_E1000_N500_full/PFNano_s-channel_mMed-900_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000  -net src/models/LGATr/lgatr.py -bs 64  --gpus 0 --run-name LGATr_training_NoPIDGL_$1_$2_$3_$4 --val-dataset-size 15000 --num-steps 200000 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --gt-radius $4 --num-blocks $1 -mv-ch $2 -s-ch $3 --spatial-part-only --validation-steps 2000 --no-pid --gen-level

exit 0
EOT

# Args: n_blocks mv_channels s_channels radius (default: 10, 16, 64, 0.8)
# bash jobs/vega/lgatr_training_vega_NoPID_on_GenParticles.sh 10 16 64 0.8
# bash jobs/vega/lgatr_training_vega_NoPID_on_GenParticles.sh 10 16 64 2.0
