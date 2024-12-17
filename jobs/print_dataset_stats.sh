source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
# Run the Python script
#python -m src.preprocessing.preprocess_dataset --input SVJ_std_UL2018_scouting_test_large --output SVJ_std_UL2018_scouting_test_large

 # Now write this with a for loop (put the 3 datasets in a list)
echo " ---- scouting_PFNano_signals/SVJ_hadronic_std ----"
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://dologarcia/gatr:v0 python -m scripts.dataset_stats --input scouting_PFNano_signals/SVJ_hadronic_std
echo " ---- scouting_PFNano_signals/SVJ_hadronic_std2 ----"
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://dologarcia/gatr:v0 python -m scripts.dataset_stats --input scouting_PFNano_signals/SVJ_hadronic_std2
echo " ---- scouting_PFNano_signals/SVJ_hadronic_std3 ----"
singularity exec -B /t3home/gkrzmanc/ -B /work/gkrzmanc/ --nv docker://dologarcia/gatr:v0 python -m scripts.dataset_stats --input scouting_PFNano_signals/SVJ_hadronic_std3

