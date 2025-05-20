# Used to store commands to download files from lxplus to another machine, e.g. PSI T3
prefix="/work/gkrzmanc/"
prefix_left="gkrzmanc@lxplus.cern.ch:/eos/user/g/gkrzmanc/jetclustering"
rsync -avz -e "ssh" --exclude "old_code" --exclude "env.sh" $prefix_left $prefix


#rsync -avz -e "ssh" t3:/work/gkrzmanc/jetclustering/preprocessed_data /eos/home-g/gkrzmanc/jetclustering/
#rsync -avz -e "ssh" t3:/work/gkrzmanc/jetclustering/results /eos/home-g/gkrzmanc/jetclustering/

# rsync -avz -e "ssh" t3:/work/gkrzmanc/jetclustering/code /ceph/hpc/home/krzmancg/jetclustering/ --exclude "wandb" --exclude ".env" --exclude "env.sh"

# rsync -avz -e "ssh" t3:/work/gkrzmanc/jetclustering/results /ceph/hpc/home/krzmancg/jetclustering/

### Vega -> T3 results
# rsync -avz -e "ssh" /ceph/hpc/home/krzmancg/jetclustering/results t3:/pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc/jetclustering

# Local to SE results
rsync -avz -e "ssh" /work/gkrzmanc/jetclustering/results/ /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc/jetclustering/results

# Local to SE data
rsync -avz -e "ssh" /work/gkrzmanc/jetclustering/data/ /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc/jetclustering/data

rsync -avz -e "ssh" /work/gkrzmanc/jetclustering/preprocessed_data/ /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc/jetclustering/preprocessed_data
rsync -avz -e "ssh"  /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc/jetclustering/preprocessed_data/ /work/gkrzmanc/jetclustering/preprocessed_data

### Local -> Vega (when T3 is down)
#rsync -avz -e "ssh -i .ssh/id_rs_sling_gk" /home/gregor/cern/jetclustering/ krzmancg@logingpu.vega.izum.si:/ceph/hpc/home/krzmancg/jetclustering/code --exclude "wandb" --exclude ".env" --exclude "env.sh" --exclude "__pycache__" --exclude ".git"

# T3 -> Vega data
# rsync -avz -e "ssh" t3:/work/gkrzmanc/jetclustering/preprocessed_data /ceph/hpc/home/krzmancg/jetclustering/
# rsync -avz -e "ssh" t3:/work/gkrzmanc/jetclustering/preprocessed_data/Feb26_2025_E1000_N500_full/PFNano_s-channel_mMed-900_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 /ceph/hpc/home/krzmancg/jetclustering/
# rsync -avz -e "ssh" t3:/work/gkrzmanc/jetclustering/preprocessed_data/Feb26_2025_E1000_N500_noPartonFilter_Folders/PFNano_s-channel_mMed-900_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 /ceph/hpc/home/krzmancg/jetclustering/preprocessed_data/Feb26_2025_E1000_N500_noPartonFilter_Folders

# T3 -> Vega code
rsync -avz -e "ssh" t3:/work/gkrzmanc/jetclustering/code /ceph/hpc/home/krzmancg/jetclustering/ --exclude "wandb" --exclude ".env" --exclude "env.sh" --exclude "__pycache__" --exclude ".git" --exclude "*.log" --exclude "*.txt"
