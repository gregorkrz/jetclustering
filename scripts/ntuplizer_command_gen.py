import argparse
import os
# This file generates cmsRun commands that saves the datasets in the appropriate folders for further processing.

# --input <input_dir_to_miniaod>
# --output <where to store the root files>

parser = argparse.ArgumentParser(description='Generate ntuples from MiniAOD')
parser.add_argument('--input', type=str, help='Input directory with MiniAOD files') # /store/user/gkrzmanc/jetclustering/sim/Feb26_2025_E1000_N500/MINIAOD
parser.add_argument("--local-prefix", type=str, default="/pnfs/psi.ch/cms/trivcat", help="Local prefix for the input files")
parser.add_argument("--prod-prefix", type=str, default="root://t3se01.psi.ch:1094", help="Prefix to use when running the script on CMS connect")
parser.add_argument('--output', type=str, help='Output directory for root files') # /store/user/gkrzmanc/jetclustering/data/26022025_E1000_N500
args = parser.parse_args()


# cmd example: cmsRun SVJScouting/test/ScoutingNanoAOD_fromMiniAOD_cfg.py inputFiles=file:root://t3se01.psi.ch:1094/store/user/gkrzmanc/jetclustering/sim/26feb/MINIAOD/step_MINIAOD_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-10_part-9.root   outputFile=out_26Feb_1.root  maxEvents=-1 isMC=true era=2018 signal=True
#print("------------------------------------------------------------------------------------------------")
for file in os.listdir(args.local_prefix+args.input):
    # filename looks like this. "step_MINIAOD_s-channel_mMed-1200_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000_part-440.root" extract the part with s-channel....n-1000
    # and use it as the output folder name
    parts = file.split("_")
    output_folder = "_".join(parts[2:-1])
    output_filename = "PFNano_"+"_".join(parts[2:])
    print("cmsRun SVJScouting/test/ScoutingNanoAOD_fromMiniAOD_cfg.py inputFiles=file:"+args.prod_prefix+args.input+"/"+file+" outputFile="+args.prod_prefix+args.output+"/"+output_folder+"/" + output_filename + " maxEvents=-1 isMC=true era=2018 signal=True > fullfile.log 2>&1 &")
#print("------------")


