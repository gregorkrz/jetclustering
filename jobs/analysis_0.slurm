#!/bin/bash
#SBATCH --partition=standard           # Specify the partition
#SBATCH --account=t3                  # Specify the account
#SBATCH --mem=10000                   # Request 10GB of memory
#SBATCH --time=00:30:00               # Set the time limit to 1 hour
#SBATCH --job-name=SVJ_analysis_0  # Name the job
#SBATCH --output=jobs/analysis_0_output.log       # Redirect stdout to a log file
#SBATCH --error=jobs/analysis_0_error.log         # Redirect stderr to a log file

source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache


singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval/AK8 --dataset-cap 100
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval/AK8_GenJets --dataset-cap 100 --jets-object genjets
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval/GATr_rinv_03_m_900 --dataset-cap 100 --eval-dir train/Test_betaPt_BC_all_datasets_2025_01_07_17_50_45 --jets-object model_jets
# GATr model trained on m=900, r inv = 0.7
# Test_betaPt_BC_all_datasets_2025_01_08_10_54_58
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/all_models_eval/GATr_rinv_07_m_900 --dataset-cap 100 --eval-dir train/Test_betaPt_BC_all_datasets_2025_01_08_10_54_58 --jets-object model_jets


# LGATr model trained on m=900, r inv = 0.3
# train/Test_LGATr_all_datasets_2025_01_08_19_27_54

#singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input scouting_PFNano_signals2/SVJ_hadronic_std --output scouting_PFNano_signals2/SVJ_hadronic_std/LGATr_rinv_03_m_900 --dataset-cap 3000 --eval-dir train/Test_LGATr_all_datasets_2025_01_08_19_27_54 --jets-object model_jets

# Run the job:
# sbatch jobs/analysis_0.slurm
