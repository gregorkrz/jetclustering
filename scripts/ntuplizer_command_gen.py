import argparse
import os

# This file generates cmsRun commands that saves the datasets in the appropriate folders for further processing.

# --input <input_dir_to_miniaod>
# --output <where to store the root files>

parser = argparse.ArgumentParser(description='Generate ntuples from MiniAOD')
parser.add_argument('--input', type=str, help='Input directory with MiniAOD files')
parser.add_argument("--local-prefix", type=str, default="/pnfs/psi.ch/cms/trivcat", help="Local prefix for the input files")
parser.add_argument("--prod-prefix", type=str, default="/work/USERNAME/jetclustering", help="Prefix to use when running the script on CMS connect")
parser.add_argument('--output', type=str, default="data/26022025_E1000_N500", help='Output directory for root files') # data/26022025_E1000_N500
parser.add_argument("--filelist", "-fl", action="store_true")
args = parser.parse_args()

filelists = {}

#print("------------------------------------------------------------------------------------------------")
for file in os.listdir(args.local_prefix+args.input):
    # filename looks like this. "step_MINIAOD_s-channel_mMed-1200_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000_part-440.root" extract the part with s-channel....n-1000
    # and use it as the output folder name
    parts = file.split("_")
    output_folder = "_".join(parts[2:-1])
    if output_folder not in filelists:
        filelists[output_folder] = set()
    output_filename = "PFNano_"+"_".join(parts[2:])
    out_dir = args.prod_prefix+"/"+args.output+"/"+output_folder
    if args.filelist:
        #print("file:root://t3se01.psi.ch:1094"+args.local_prefix+args.input+"/"+file)
        fname = os.path.join(args.local_prefix+ args.input, file)
        #print(args.local_prefix)
        print(fname)
        filelists[output_folder].add(fname)
    else:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        print("cmsRun PhysicsTools/SVJScouting/test/ScoutingNanoAOD_fromMiniAOD_cfg.py inputFiles=" + args.input+"/"+file+" outputFile=" + args.prod_prefix + "/" + args.output + "/" + output_folder + "/" + output_filename + " maxEvents=-1 isMC=true era=2018 signal=True")

if args.filelist:
    import pickle
    pickle.dump(filelists, open("filelist.pkl", "wb"))

