#!/bin/bash
#SBATCH --partition=standard           # Specify the partition
#SBATCH --account=t3                  # Specify the account
#SBATCH --mem=10000                   # Request 10GB of memory
#SBATCH --time=01:00:00               # Set the time limit to 1 hour
#SBATCH --job-name=SVJeval  # Name the job
#SBATCH --output=jobs/eval_models.log       # Redirect stdout to a log file
#SBATCH --error=jobs/eval_models_err.log         # Redirect stderr to a log file
#SBATCH -n 2
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache


#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval/AK8 &
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval/AK8_GenJets --jets-object genjets &
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval/GATr_rinv_03_m_900  --eval-dir train/Test_betaPt_BC_all_datasets_2025_01_07_17_50_45 --jets-object model_jets &
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval/GATr_rinv_07_m_900  --eval-dir train/Test_betaPt_BC_all_datasets_2025_01_08_10_54_58 --jets-object model_jets &
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval/GATr_rinv_03_m_900_C1010  --eval-dir train/Test_betaPt_BC_all_datasets_2025_01_07_17_50_45 --jets-object model_jets --clustering-suffix 1010  &
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval/GATr_rinv_07_m_900_C1010  --eval-dir train/Test_betaPt_BC_all_datasets_2025_01_08_10_54_58 --jets-object model_jets --clustering-suffix 1010  &
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval/LGATr_rinv_03_m_900 --eval-dir train/Eval_LGATr_SB_spatial_part_only_1_2025_01_13_14_31_58 --jets-object model_jets  &


# Identity eval
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval/identity  --eval-dir train/Eval_Identity_2025_01_13_17_08_48 --jets-object model_jets  &
# Identity eval with FT clustering
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_clustering/identity  --eval-dir train/Eval_Identity_2025_01_13_17_08_48 --jets-object model_jets --clustering-suffix FT  &

# Eval of step13k L-GATr #Eval_LGATr_SB_spatial_part_only_1_step_13k_2025_01_13_17_28_03
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval/LGATr_rinv_03_m_900_S13k --eval-dir train/Eval_LGATr_SB_spatial_part_only_1_step_13k_2025_01_13_17_28_03 --jets-object model_jets  &
# FT clustering
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_clustering/LGATr_rinv_03_m_900_S13k --eval-dir train/Eval_LGATr_SB_spatial_part_only_1_step_13k_2025_01_13_17_28_03 --jets-object model_jets --clustering-suffix FT   &
# Ft clustering + 40k L-GATr model
# Eval_LGATr_SB_spatial_part_only_1_2025_01_13_14_31_58
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_clustering/LGATr_rinv_03_m_900 --eval-dir train/Eval_LGATr_SB_spatial_part_only_1_2025_01_13_14_31_58 --jets-object model_jets --clustering-suffix FT   &


### Eval finetuning

#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R/LGATr_R06 --eval-dir train/Eval_GT_R_lgatr_R06_2025_01_16_13_41_48 --jets-object model_jets    &
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R/LGATr_R10 --eval-dir train/Eval_GT_R_lgatr_R10_2025_01_16_13_41_52 --jets-object model_jets    &
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R/LGATr_R07 --eval-dir train/Eval_GT_R_lgatr_R07_2025_01_16_13_41_41 --jets-object model_jets    &
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R/LGATr_R09 --eval-dir train/Eval_GT_R_lgatr_R09_2025_01_16_13_41_45 --jets-object model_jets    &
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_clustering/LGATr_train_all_26k --eval-dir train/Eval_all_train_26k_2025_01_16_13_41_37 --jets-object model_jets   &


# Eval transformer baseline
# "/work/gkrzmanc/jetclustering/results/train/Eval_Transformer_2025_01_17_15_27_59"
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_clustering/Eval_Transformer --eval-dir train/Eval_Transformer_2025_01_17_15_27_59 --jets-object model_jets  &



# Eval runs from 18 January
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_clustering/LGATr_scalar_coords --eval-dir train/Eval_lgatr_scalars_OC_2025_01_18_15_28_01 --jets-object model_jets  &
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_clustering/LGATr_no_coords_loss --eval-dir train/Eval_LGATr_no_coords_loss_1_2025_01_18_15_21_00 --jets-object model_jets  &
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R/LGATr_R14 --eval-dir train/Eval_GT_R_lgatr_R14_2025_01_18_13_28_47 --jets-object model_jets  &
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R/LGATr_R12 --eval-dir train/Eval_GT_R_lgatr_R12_2025_01_18_13_28_44 --jets-object model_jets  &
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R/LGATr_R11 --eval-dir train/Eval_GT_R_lgatr_R11_2025_01_18_13_27_02 --jets-object model_jets  &


################

wait
echo "All scripts done"

# LGATr model trained on m=900, r inv = 0.3
# train/Test_LGATr_all_datasets_2025_01_08_19_27_54

#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/LGATr_rinv_03_m_900 --dataset-cap 3000 --eval-dir train/Test_LGATr_all_datasets_2025_01_08_19_27_54 --jets-object model_jets

# Run the job:
# sbatch jobs/eval_models.slurm
