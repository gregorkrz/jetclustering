#!/bin/bash
#SBATCH --partition=standard           # Specify the partition
#SBATCH --account=t3                  # Specify the account
#SBATCH --mem=10000                   # Request 10GB of memory
#SBATCH --time=02:00:00              # Set the time limit to 1 hour
#SBATCH --job-name=SVJeval  # Name the job
#SBATCH --output=jobs/eval_models21.log       # Redirect stdout to a log file
#SBATCH --error=jobs/eval_models_err21.log         # Redirect stderr to a log file
#SBATCH -n 1
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache




#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R/AK8 &
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R/AK8_GenJets --jets-object genjets &
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
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R/LGATr_rinv_03_m_900 --eval-dir train/Eval_LGATr_SB_spatial_part_only_1_2025_01_13_14_31_58 --jets-object model_jets --clustering-suffix FT   &


### Eval finetuning

#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R/LGATr_R06 --eval-dir train/Eval_GT_R_lgatr_R06_2025_01_16_13_41_48 --jets-object model_jets    &
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R/LGATr_R10 --eval-dir train/Eval_GT_R_lgatr_R10_2025_01_16_13_41_52 --jets-object model_jets    &
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R/LGATr_R07 --eval-dir train/Eval_GT_R_lgatr_R07_2025_01_16_13_41_41 --jets-object model_jets    &
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R/LGATr_R09 --eval-dir train/Eval_GT_R_lgatr_R09_2025_01_16_13_41_45 --jets-object model_jets &
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R/LGATr_R08 --eval-dir train/Eval_LGATr_SB_spatial_part_only_1_step_13k_2025_01_13_17_28_03 --jets-object model_jets --clustering-suffix FT   &

#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_clustering/LGATr_train_all_26k --eval-dir train/Eval_all_train_26k_2025_01_16_13_41_37 --jets-object model_jets   &



# Eval transformer baseline
# "/work/gkrzmanc/jetclustering/results/train/Eval_Transformer_2025_01_17_15_27_59"
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_clustering/Transformer_S26k --eval-dir train/Eval_Transformer_2025_01_17_15_27_59 --jets-object model_jets


#Eval_Transformer_2025_01_20_11_49_05
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_clustering/Transformer_S40k --eval-dir train/Eval_Transformer_2025_01_20_11_49_05 --jets-object model_jets

# Eval_Transformer_2025_01_20_13_37_45
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_clustering/Transformer_S170k --eval-dir train/Eval_Transformer_2025_01_20_11_49_05 --jets-object model_jets  &


# Eval runs from 18 January
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_clustering/LGATr_scalar_coords --eval-dir train/Eval_lgatr_scalars_OC_2025_01_18_15_28_01 --jets-object model_jets  &
#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_clustering/LGATr_no_coords_loss --eval-dir train/Eval_LGATr_no_coords_loss_1_2025_01_18_15_21_00 --jets-object model_jets  &
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R/LGATr_R14 --eval-dir train/Eval_GT_R_lgatr_R14_2025_01_18_13_28_47 --jets-object model_jets
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R/LGATr_R12 --eval-dir train/Eval_GT_R_lgatr_R12_2025_01_18_13_28_44 --jets-object model_jets
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R/LGATr_R11 --eval-dir train/Eval_GT_R_lgatr_R11_2025_01_18_13_27_02 --jets-object model_jets
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R/LGATr_R25 --eval-dir train/Eval_GT_R_lgatr_R25_2025_01_22_10_51_36 --jets-object model_jets
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R/LGATr_R20 --eval-dir train/Eval_GT_R_lgatr_R20_2025_01_22_10_51_30 --jets-object model_jets



# TODO rerun with SP1
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_R/LGATr_R14_SP --eval-dir train/Eval_GT_R_lgatr_R14_2025_01_18_13_28_47 --jets-object model_jets --clustering-suffix SP1  &
# No coords loss: Eval_LGATr_no_coords_loss_1_2025_01_20_16_22_21
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_clustering/LGATr_no_coords_loss --eval-dir train/Eval_LGATr_no_coords_loss_1_2025_01_20_16_22_21 --jets-object model_jets

# Eval Eval_Quark_dist_loss_2025_01_18_13_11_16
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_clustering/LGATr_quark_dist_cos_sim_loss --eval-dir train/Eval_Quark_dist_loss_2025_01_18_13_11_16 --jets-object model_jets --clustering-suffix FT1 &

#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_ds_cap/Transformer_S40K_Train5k --eval-dir train/transformer_eval_ds_cap_DS5000_2025_01_22_10_58_13 --jets-object model_jets  &

#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_ds_cap/Transformer_S40K_Train1k --eval-dir train/transformer_eval_ds_cap_DS1000_2025_01_22_10_58_17 --jets-object model_jets  &


# transformer eval S15k
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_ds_cap/Transformer_S15K_Train5k --eval-dir train/transformer_eval_ds_cap_DS5000_s15k_2025_01_22_11_16_53 --jets-object model_jets  &
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_ds_cap/Transformer_S15K_Train1k --eval-dir train/transformer_eval_ds_cap_DS1000_s15k_2025_01_22_11_48_41 --jets-object model_jets  &
#lgatr_eval_ds_cap_5000_S15k_2025_01_22_13_03_41
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_ds_cap/LGATr_S15K_Train5k --eval-dir train/lgatr_eval_ds_cap_5000_S15k_2025_01_22_13_03_41 --jets-object model_jets  &
#lgatr_eval_ds_cap_1000_S15k_2025_01_22_13_03_38
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_ds_cap/LGATr_S15K_Train1k --eval-dir train/lgatr_eval_ds_cap_1000_S15k_2025_01_22_13_03_38 --jets-object model_jets  &
#lgatr_eval_ds_cap_5000_S40k_2025_01_22_15_41_30
#srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_ds_cap/LGATr_S40K_Train5k --eval-dir train/lgatr_eval_ds_cap_5000_S40k_2025_01_22_15_41_30 --jets-object model_jets  &
srun singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval_FT_ds_cap/LGATr_S40K_Train1k --eval-dir train/lgatr_eval_ds_cap_1000_S40k_2025_01_22_17_13_59 --jets-object model_jets  &

################

wait
echo "All scripts done"


# LGATr model trained on m=900, r inv =quark_dist 0.3
# train/Test_LGATr_all_datasets_2025_01_08_19_27_54

#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/LGATr_rinv_03_m_900 --dataset-cap 3000 --eval-dir train/Test_LGATr_all_datasets_2025_01_08_19_27_54 --jets-object model_jets

# Run the job:
# sbatch jobs/eval_models.slurm
