export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
singularity shell  -B /work/gkrzmanc/ --nv docker://dologarcia/gatr:v0

