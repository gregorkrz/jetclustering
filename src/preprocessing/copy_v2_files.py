# The files from the ntuplizer are produced in parts, so we need to copy them to the appropriate folders in order to have one dataset per signal hypothesis.


import argparse
import os
parser = argparse.ArgumentParser(description='Copy the files in appropriate folders - no matter how many parts are in the files')
parser.add_argument("--input", type=str, default="/pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc/jetclustering/data/Feb26_2025_E1000_N500")
parser.add_argument("--output", type=str, default="/pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc/jetclustering/data/Feb26_2025_E1000_N500_folders")
parser.add_argument("--overwrite", action="store_true") # if true, it will overwrite the files, otherwise, it will skip files that have been already copied

args = parser.parse_args()

if not os.path.exists(args.output):
    os.makedirs(args.output)

for file in os.listdir(args.input):
    # if file is less than 0.5MB, ignore - it's probably still in processing
    if os.path.getsize(os.path.join(args.input, file)) < 0.5e6:
        continue
    # filename is like this: PFNano_s-channel_mMed-1100_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000_part_1.root
    # make a dir PFNano_s-channel_mMed-1100_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000 in output
    # and put part_X.root in it
    filename = file.split("_part_")[0]
    if not os.path.exists(os.path.join(args.output, filename)):
        os.makedirs(os.path.join(args.output, filename))
    # copy it
    print(f"Copying {file} to {os.path.join(args.output, filename)}")
    if args.overwrite or not os.path.exists( os.path.join(args.output, filename, "part_"+file.split("_part_")[1])):
        os.system(f"cp {os.path.join(args.input, file)} {os.path.join(args.output, filename)}")
        # rename it
        os.rename(os.path.join(args.output, filename, file), os.path.join(args.output, filename, "part_"+file.split("_part_")[1]))

print("Done")
