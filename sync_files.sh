# Used to store commands to download files from lxplus to another machine, e.g. PSI T3
prefix="/work/gkrzmanc/"
prefix_left="gkrzmanc@lxplus.cern.ch:/eos/user/g/gkrzmanc/jetclustering"
rsync -avz -e "ssh" --exclude "old_code" --exclude "env.sh" $prefix_left $prefix


#rsync -avz -e "ssh" t3:/work/gkrzmanc/jetclustering/preprocessed_data /eos/home-g/gkrzmanc/jetclustering/
#rsync -avz -e "ssh" t3:/work/gkrzmanc/jetclustering/results /eos/home-g/gkrzmanc/jetclustering/

# rsync -avz -e "ssh" t3:/work/gkrzmanc/jetclustering/code /ceph/hpc/home/krzmancg/jetclustering/ --exclude "wandb" --exclude ".env" --exclude "env.sh"

# rsync -avz -e "ssh" t3:/work/gkrzmanc/jetclustering/results /ceph/hpc/home/krzmancg/jetclustering/


### Vega -> T3
# rsync -avz -e "ssh" /ceph/hpc/home/krzmancg/jetclustering/results t3:/work/gkrzmanc/jetclustering
