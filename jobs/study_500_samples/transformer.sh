#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpu               # Specify the partition
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=48:00:00
#SBATCH --job-name=DelphesSVJTrain  # Name the job
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres             # Specify the account
#SBATCH --output=jobs/vega/DTr_out_$1_$2_$3_R$4.log
#SBATCH --error=jobs/vega/DTr_err_$1_$2_$3_R$4.log

source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi

srun singularity exec  -B /work -B /pnfs -B /t3home --nv docker://gkrz/lgatr:v3 python -m src.train -train Delphes_020425_train2_PU_PFfix_part0/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_PU_PFfix_part1/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_PU_PFfix_part2/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_PU_PFfix_part3/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_PU_PFfix_part4/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_PU_PFfix_part5/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_PU_PFfix_part6/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak  Delphes_020425_train2_PU_PFfix_part7/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak  Delphes_020425_train2_PU_PFfix_part8/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak  -val Delphes_020425_train2_PU_PFfix_part9/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak -net src/models/transformer/transformer.py -bs 20  --gpus 0 --run-name Transformer_training_NoPID_Delphes_PU_CoordFix_SmallDS_$1_$2_$3_$4 --val-dataset-size 1000 --num-steps 2000 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --gt-radius $4 --num-blocks $1 -mv-ch $2 -s-ch $3 --spatial-part-only --validation-steps 50 --no-pid  --train-dataset-size 500

exit 0
EOT

# Args: n_blocks mv_channels s_channels radius (default: 10, 16, 64, 0.8)
# bash jobs/study_500_samples/transformer.sh 10 16 64 0.8

