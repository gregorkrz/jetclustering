#!/bin/bash
#SBATCH --partition=standard           # Specify the partition
#SBATCH --account=t3                  # Specify the account
#SBATCH --mem=10000                   # Request 10GB of memory
#SBATCH --time=05:00:00               # Set the time limit to 1 hour
#SBATCH --job-name=SVJ_clustering  # Name the job
#SBATCH --output=jobs/clustering_out.log       # Redirect stdout to a log file
#SBATCH --error=jobs/clustering_err.log         # Redirect stderr to a log file
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --input train/Test_betaPt_BC_all_datasets_2025_01_07_17_50_45 --output-suffix 1010 --min-cluster-size 10 --min-samples 10

# Eval of the model trained on m=900 rinv=0.7
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --input train/Test_betaPt_BC_all_datasets_2025_01_08_10_54_58 --output-suffix 1010 --min-cluster-size 10 --min-samples 10

# Clustering eval on the L-GATr
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --input train/Test_LGATr_all_datasets_2025_01_08_19_27_54


# Finetuned parameters for L-GATr_m_900_rinv_0.3
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --input train/Eval_LGATr_SB_spatial_part_only_1_step_13k_2025_01_13_17_28_03 --output-suffix FT --min-cluster-size 11 --min-samples 18 --epsilon 0.48

# Finetuned parameters for Identity
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --input train/Eval_Identity_2025_01_13_17_08_48 --output-suffix FT --min-cluster-size 9 --min-samples 2 --epsilon 0.051


# Finetuned parameters for L-GATr_m_900_rinv_0.3 ran on the model at 40k+ steps (not 13k) Eval_LGATr_SB_spatial_part_only_1_2025_01_13_14_31_58
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --input train/Eval_LGATr_SB_spatial_part_only_1_2025_01_13_14_31_58 --output-suffix FT --min-cluster-size 11 --min-samples 18 --epsilon 0.48

srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --input train/Eval_GT_R_lgatr_R14_2025_01_18_13_28_47 --output-suffix SP --min-cluster-size 11 --min-samples 18 --epsilon 0.48



# Run the job:
# sbatch jobs/clustering.slurm
