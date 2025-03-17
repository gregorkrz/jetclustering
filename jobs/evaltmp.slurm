#!/bin/bash
#SBATCH --partition=gpu           # Specify the partition
#SBATCH --account=gpu_gres               # Specify the account
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=02:00:00               # Set the time limit to 1 hour
#SBATCH --job-name=SVJeval  # Name the job
#SBATCH --error=jobs/logs/no_pid_eval_full_12_err.txt       # Redirect stderr to a log file
#SBATCH --output=jobs/logs/no_pid_eval_full_12_log.txt      # Redirect stderr to a log file
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gkrzmanc@student.ethz.ch
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp1
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ -B /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc/  --nv docker://gkrz/lgatr:v3 python -m src.train -test  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000 --gpus 0 --run-name Eval_no_pid_eval_full --load-model-weights /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc/jetclustering/results/train/LGATr_training_NoPID_10_16_64_0.8_AllData_2025_02_28_13_42_59/step_40000_epoch_2.ckpt --num-workers 0  --tag no_pid_eval_full --load-from-run LGATr_training_NoPID_10_16_64_0.8_AllData_2025_02_28_13_42_59 --ckpt-step 40000    --parton-level  --epsilon 0.5 --min-samples 2 --min-cluster-size 4

