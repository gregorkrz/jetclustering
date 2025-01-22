#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpu               # Specify the partition
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=40:00:00
#SBATCH --job-name=SVJTransformerTraining  # Name the job
#SBATCH --gres=gpu:1
#SBATCH --account=s25t01-01-users            # Specify the account
#SBATCH --output=jobs/vega/training_transformer_vega_out_$1_$2_$3.log
#SBATCH --error=jobs/vega/training_transformer_vega_err_$1_$2_$3.log

source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi

srun singularity exec -B /ceph/hpc/home/krzmancg --nv docker://gkrz/lgatr:v3 python -m src.train -train scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-900_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 -val scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-900_mDark-20_rinv-0.3 -net src/models/transformer/transformer.py -bs 64  --gpus 0 --run-name Transformer_training_40k_$1_$2_$3 --val-dataset-size 1024 --num-steps 40000 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --gt-radius 0.8 --num-blocks $1 --internal-dim $2 --n-heads $3

exit 0
EOT


# Args: n_blocks internal_dim n_heads (default: 10, 128, 4) - >gives around 1 millioon parameters
# bash jobs/vega/transformer_training_vega.sh 3 32 4 -> gives around 25k parameters
# bash jobs/vega/transformer_training_vega.sh 5 32 4

