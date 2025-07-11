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

#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --input train/Eval_GT_R_lgatr_R14_2025_01_18_13_28_47 --output-suffix SP1 --min-cluster-size 11 --min-samples 18 --epsilon 0.48 --spatial-part-only

# Run
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --input train/Eval_Quark_dist_loss_2025_01_18_13_11_16 --output-suffix FT1 --min-cluster-size 8 --min-samples 3 --epsilon 0.09 --spatial-part-only


# No coords loss
# {'min_cluster_size': 5, 'min_samples': 19, 'epsilon': 0.17149658495077644}
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_NoQMinReprod_2025_04_04_11_13_28_127 --output-suffix FT1 --min-cluster-size 2 --min-samples 1 --epsilon 0.3 --spatial-part-only
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_NoQMinReprod_2025_04_04_11_13_23_90 --output-suffix FT1 --min-cluster-size 2 --min-samples 1 --epsilon 0.3 --spatial-part-only
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_NoQMinReprod_2025_04_04_11_13_20_253 --output-suffix FT1 --min-cluster-size 2 --min-samples 1 --epsilon 0.3 --spatial-part-only
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_NoQMinReprod_2025_04_04_11_13_18_71 --output-suffix FT1 --min-cluster-size 2 --min-samples 1 --epsilon 0.3 --spatial-part-only
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_NoQMinReprod_2025_04_04_10_32_06_532 --output-suffix FT1 --min-cluster-size 2 --min-samples 1 --epsilon 0.3 --spatial-part-only
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_NoQMinReprod_2025_04_04_10_32_03_809 --output-suffix FT1 --min-cluster-size 2 --min-samples 1 --epsilon 0.3 --spatial-part-only
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_NoQMinReprod_2025_04_04_10_32_00_500 --output-suffix FT1 --min-cluster-size 2 --min-samples 1 --epsilon 0.3 --spatial-part-only
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_NoQMinReprod_2025_04_04_10_31_57_566 --output-suffix FT1 --min-cluster-size 2 --min-samples 1 --epsilon 0.3 --spatial-part-only
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_NoQMinReprod_2025_04_04_10_31_53_462 --output-suffix FT1 --min-cluster-size 2 --min-samples 1 --epsilon 0.3 --spatial-part-only
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.compute_clustering --input train/Eval_eval_19March2025_pt1e-2_500particles_NoQMinReprod_2025_04_04_10_31_51_450 --output-suffix FT1 --min-cluster-size 2 --min-samples 1 --epsilon 0.3 --spatial-part-only

# Run the job:

# sbatch jobs/clustering.slurm
