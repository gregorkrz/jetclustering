#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpu               # Specify the partition
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=40:00:00
#SBATCH --job-name=SVJTransformerTraining  # Name the job
#SBATCH --gres=gpu:1
#SBATCH --account=s25t01-01-users             # Specify the account
#SBATCH --output=jobs/vega/training_gatr_out_$1_$2_$3.log
#SBATCH --error=jobs/vega/training_gatr_err_$1_$2_$3.log

source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi

srun singularity exec  -B /ceph/hpc/home/krzmancg --nv docker://gkrz/lgatr:v3 python -m src.train -train scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-900_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 -val scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-700_mDark-20_rinv-0.7 -net src/models/GATr/Gatr.py -bs 64  --gpus 0 --run-name GATr_training_$1_$2_$3 --val-dataset-size 15000 --num-steps 100000 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --gt-radius 0.8 --num-blocks $1 -mv-ch $2 -s-ch $3 --validation-steps 2000

exit 0
EOT


# Args: n_blocks mv_channels s_channels (default: 10, 16, 64)
# bash jobs/vega/gatr_training_vega.sh 2 4 4
# bash jobs/vega/gatr_training_vega.sh 3 4 4
# bash jobs/vega/gatr_training_vega.sh 5 4 16
