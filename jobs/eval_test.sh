#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpu               # Specify the partition
#SBATCH --account=gpu_gres            # Specify the account
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=05:00:00
#SBATCH --job-name=SVJtesteval  # Name the job
#SBATCH --gres=gpu:1
#SBATCH --output=jobs/eval_${SLURM_JOB_ID}.out
#SBATCH --error=jobs/eval_${SLURM_JOB_ID}.err
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi

srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ -B /pnfs --nv docker://gkrz/lgatr:v3 python -m src.train -test Feb26_2025_E1000_N500_full/PFNano_s-channel_mMed-1100_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000 --gpus 0 --run-name Eval_no_pid_eval_full_1 --load-model-weights /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc/jetclustering/results/train/LGATr_training_NoPID_10_16_64_2.0_2025_02_28_12_48_58/step_40000_epoch_30.ckpt --num-workers 0 --tag debugging --load-from-run LGATr_training_NoPID_10_16_64_2.0_2025_02_28_12_48_58 --ckpt-step 40000 --parton-level --epsilon 0.5 --min-samples 2 --min-cluster-size 4

exit
EOT



# Big training eval
#  bash jobs/eval_test.sh


