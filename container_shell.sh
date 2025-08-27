export APPTAINER_TMPDIR=/work/USER/singularity_tmp
export APPTAINER_CACHEDIR=/work/USER/singularity_cache
singularity shell  -B /work/USER/ --nv docker://<CONTAINER_NAME>
