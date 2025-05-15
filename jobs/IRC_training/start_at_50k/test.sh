#!/bin/bash
sbatch <<EOT
  #!/bin/bash
  #SBATCH --partition=short               # Specify the partition
  #SBATCH --mem=25000                   # Request 10GB of memory
  #SBATCH --time=00:01:00
  #SBATCH --job-name=test1  # Name the job
  #SBATCH --account=t3            # Specify the account
  #SBATCH --output=jobs/vega/Dtr_lgatr_aug.out
  #SBATCH --error=jobs/vega/Dtr_lgatr_aug.err

  source env.sh
  export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
  export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache

nvidia-smi
env
echo " ---- end env ---- "
srun singularity exec  -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ -B /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc -H /t3home/gkrzmanc  --nv docker://gkrz/lgatr:v3 python -m pip install  --no-input fastjet

srun singularity exec  -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ -B /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc -H /t3home/gkrzmanc    --nv docker://gkrz/lgatr:v3 python -c "import fastjet"
echo "Hello"
srun singularity exec  -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ -B /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc  -H /t3home/gkrzmanc   --nv docker://gkrz/lgatr:v3 python -m src.train  -train Delphes_020425_train2_PU_PFfix_part0/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_PU_PFfix_part1/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_PU_PFfix_part2/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_PU_PFfix_part3/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_PU_PFfix_part4/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_PU_PFfix_part5/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak Delphes_020425_train2_PU_PFfix_part6/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak  Delphes_020425_train2_PU_PFfix_part7/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak  Delphes_020425_train2_PU_PFfix_part8/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak  -val Delphes_020425_train2_PU_PFfix_part9/SVJ_mZprime-900_mDark-20_rinv-0.3_alpha-peak  -net src/models/LGATr/lgatr.py -bs 8 --gpus 0 --run-name Delphes_NOAug_IRCSplit_50k_cont --val-dataset-size 150 --num-steps 200000 --attr-loss-weight 0.1 --coord-loss-weight 0.1 --beta-type pt+bc --gt-radius 0.8 --num-blocks 10 -mv-ch 16 -s-ch 64 --spatial-part-only --validation-steps 60 --no-pid --load-model-weights  train/Delphes_NOAug_IRCSplit_50k__2025_05_13_09_56_39_345/step_2460_epoch_1.ckpt   --num-workers 0 -irc

exit 0
EOT

# bash jobs/IRC_training/start_at_50k/test.sh
####



