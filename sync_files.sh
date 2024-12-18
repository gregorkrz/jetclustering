# Used to store commands to download files from lxplus to another machine, e.g. PSI T3
prefix="/work/gkrzmanc/"
prefix_left="gkrzmanc@lxplus.cern.ch:/eos/user/g/gkrzmanc/jetclustering"
rsync -avz -e "ssh" --exclude "old_code" --exclude "env.sh" $prefix_left $prefix


#rsync -avz -e "ssh" gkrzmanc@t3ui01.psi.ch:/work/gkrzmanc/jetclustering/preprocessed_data /eos/home-g/gkrzmanc/jetclustering/
#rsync -avz -e "ssh" gkrzmanc@t3ui01.psi.ch:/work/gkrzmanc/jetclustering/results /eos/home-g/gkrzmanc/jetclustering/

