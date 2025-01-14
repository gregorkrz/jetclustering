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



### Local -> Vega (when T3 is down)
#rsync -avz -e "ssh -i .ssh/id_rs_sling_gk" /home/gregor/cern/jetclustering/ krzmancg@logingpu.vega.izum.si:/ceph/hpc/home/krzmancg/jetclustering/code --exclude "wandb" --exclude ".env" --exclude "env.sh" --exclude "__pycache__" --exclude ".git"


# T3 -> Vega data
# rsync -avz -e "ssh" t3:/work/gkrzmanc/jetclustering/preprocessed_data/scouting_PFNano_signals2 /ceph/hpc/home/krzmancg/jetclustering/

# T3 -> Vega code
rsync -avz -e "ssh" t3:/work/gkrzmanc/jetclustering/code /ceph/hpc/home/krzmancg/jetclustering/ --exclude "wandb" --exclude ".env" --exclude "env.sh" --exclude "__pycache__" --exclude ".git"


