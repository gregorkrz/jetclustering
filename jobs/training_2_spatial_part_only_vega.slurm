#!/bin/bash
#SBATCH --job-name="SVJtr2b"
#SBATCH --time=25:10:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=gpu
#SBATCH --mem=25GB
#SBATCH --output=jobs/training_2_output2b.log
#SBATCH --error=jobs/training_2_error2b.log

source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi

# Training datasets:
# Validation datasets:

#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3  python -m src.train -train scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-900_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-2000 -val scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-900_mDark-20_rinv-0.3 -net src/models/LGATr/lgatr.py -bs 256 --gpus 0 --run-name LGATr_train_full --val-dataset-size 256 --num-epochs 1000 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --spatial-part-only
srun singularity exec -B /ceph/hpc/home/krzmancg --nv docker://gkrz/lgatr:v3  python -m src.train -train scouting_PFNano_signals2/SVJ_hadronic_std3/s-channel_mMed-900_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-2000 -val scouting_PFNano_signals2/SVJ_hadronic_std2/s-channel_mMed-900_mDark-20_rinv-0.7 -net src/models/LGATr/lgatr.py -bs 256 --gpus 0 --run-name LGATr_train_full_rinv07 --val-dataset-size 2000 --num-epochs 1000 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --spatial-part-only --num-workers 0

# sbatch jobs/training_2_spatial_part_only_vega.slurm
