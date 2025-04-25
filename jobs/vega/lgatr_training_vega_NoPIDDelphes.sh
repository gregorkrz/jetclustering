#!/bin/bash
sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpu               # Specify the partition
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=48:00:00
#SBATCH --job-name=DelphesSVJTrain  # Name the job
#SBATCH --gres=gpu:1
#SBATCH --account=gpu_gres             # Specify the account
#SBATCH --output=jobs/vega/DTrLGATr_out_$1_$2_$3_R$4.log
#SBATCH --error=jobs/vega/DTrLGATr_err_$1_$2_$3_R$4.log


source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi

srun singularity exec  -B /work -B /pnfs -B /t3home --nv docker://gkrz/lgatr:v3 python -m src.train -train Delphes_020425_train2_part0/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_part1/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_part2/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_part3/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_part4/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_part5/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_part6/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak  Delphes_020425_train2_part7/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak  Delphes_020425_train2_part8/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak  -val Delphes_020425_train2_part9/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak -net src/models/LGATr/lgatr.py -bs 64  --gpus 0 --run-name LGATr_training_NoPID_Delphes_$1_$2_$3_$4 --val-dataset-size 1000 --num-steps 200000 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --gt-radius $4 --num-blocks $1 -mv-ch $2 -s-ch $3 --spatial-part-only --validation-steps 2000 --no-pid

exit 0
EOT


# Args: n_blocks mv_channels s_channels radius (default: 10, 16, 64, 0.8)
# bash jobs/vega/lgatr_training_vega_NoPIDDelphes.sh 10 16 64 0.8
# bash jobs/vega/lgatr_training_vega_NoPIDDelphes.sh 10 16 64 2.0

