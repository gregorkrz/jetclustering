#!/bin/bash
#SBATCH --partition=gpu           # Specify the partition
#SBATCH --account=gpu_gres                  # Specify the account
#SBATCH --mem=500M                  # Request 10GB of memory
#SBATCH --time=00:00:30             # Set the time limit to 1 hour
#SBATCH --job-name=GPUinfo  # Name the job
#SBATCH --output=jobs/GPUstats_output.log       # Redirect stdout to a log file
#SBATCH --error=jobs/GPUstats_error.log         # Redirect stderr to a log file
#SBATCH --gres=gpu:1

nvidia-smi

# Run the job:
# sbatch jobs/gpustats.slurm
