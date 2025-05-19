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

srun singularity exec  -B /work -B /pnfs -B /t3home -H /t3home/gkrzmanc --nv docker://gkrz/lgatr:v3 python -m src.train -train QCDtrain_part0/qcd_test_0 QCDtrain_part0/qcd_test_1 QCDtrain_part0/qcd_test_2 QCDtrain_part0/qcd_test_3 QCDtrain_part0/qcd_test_4 QCDtrain_part0/qcd_test_5 QCDtrain_part0/qcd_test_6 QCDtrain_part0/qcd_test_7 QCDtrain_part0/qcd_test_8  -val QCDtrain_part0/qcd_test_9 -net src/models/LGATr/lgatr.py -bs 20  --gpus 0 --run-name GP_IRC_S_LGATr_training_NoPID_Delphes_PU_PFfix_QCD_events_$1_$2_$3_$4 --val-dataset-size 1000 --num-steps 200000 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --gt-radius $4 --num-blocks $1 -mv-ch $2 -s-ch $3 --spatial-part-only --validation-steps 2000 --no-pid --augment-soft-particles  --load-model-weights train/LGATr_training_NoPID_Delphes_PU_PFfix_QCD_events_10_16_64_0.8_2025_05_16_19_46_57_48/step_50000_epoch_12.ckpt -irc

exit 0
EOT

# Args: n_blocks mv_channels s_channels radius (default: 10, 16, 64, 0.8)
# bash jobs/base_training_different_datasets/aug_IRC_S/lgatr_QCD.sh 10 16 64 0.8
