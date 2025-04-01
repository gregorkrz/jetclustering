# Eval a given a training run name at the given steps, taking into account the chaning of the training runs

import sys
import pickle
import wandb
import argparse
import os

from src.utils.paths import get_path
from src.utils.wandb_utils import get_run_initial_steps, get_run_step_direct, get_run_step_ckpt, get_steps_from_file, get_run_by_name

parser = argparse.ArgumentParser()
parser.add_argument("--tag", "-tag", type=str, required=False, default="")
parser.add_argument("--input", "-input", type=str, required=False, default="Feb26_2025_E1000_N500_noPartonFilter_C_F") # --input Feb26_2025_E1000_N500_full
parser.add_argument("--clustering-suffix", "-c", type=str, required=False, default="") #  -c MinSamples0
parser.add_argument("--no-submit", "-ns", action="store_true") # do not submit the slurm job
parser.add_argument("--submit-AKX", "-AKX", action="store_true")
parser.add_argument("--submit-AK8", "-AK8", action="store_true")
parser.add_argument("--parton-level", "-pl", action="store_true") # To be used together with 'fastjet_jets' and --submit-AKX
parser.add_argument("--gen-level", "-gl", action="store_true")

args = parser.parse_args()
api = wandb.Api()

DSCAP = 2000

def get_eval_run_names(tag):
    # from the api, get all the runs with the tag that are finished
    runs = api.runs(
        path="fcc_ml/svj_clustering",
        filters={"tags": {"$in": [tag.strip()]}}
    )
    return [run.name for run in runs if run.state == "finished"], [run.config for run in runs if run.state == "finished"]

def get_log_number(tag):
    numbers = set()
    for file in os.listdir("jobs/slurm_files"):
        if tag in file:
            numbers.add(int(file.split("_")[-1].split(".")[0]))
    if len(numbers) == 0:
        return 0
    return max(list(numbers)) + 1

def get_slurm_file_text_AKX(tag, log_number):
    bindings = "-B /t3home/gkrzmanc/ -B /work/gkrzmanc/"
    partition = "standard"
    account = "t3"
    d = "jobs/logs/{}".format(tag)
    err = d + "_{}_CPUerr.txt".format(log_number)
    log = d + "_{}_CPUlog.txt".format(log_number)
    suffix_pl = "--parton-level" if args.parton_level else ""
    suffix_gl = "--gen-level" if args.gen_level else ""
    pl_folder = "_PL" if args.parton_level else ""
    gl_folder = "_GL" if args.gen_level else ""
    file = f"""#!/bin/bash
#SBATCH --partition={partition}           # Specify the partition
#SBATCH --account={account}                  # Specify the account
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=06:00:00               # Set the time limit to 1 hour
#SBATCH --job-name=SVJan_AKX{pl_folder}{gl_folder}  # Name the job
#SBATCH --error={err}         # Redirect stderr to a log file
#SBATCH --output={log}         # Redirect stderr to a log file
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi
srun singularity exec {bindings} docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input {args.input} --output {args.input}/batch_eval_2k/{tag}/AKX{pl_folder}{gl_folder} --jets-object fastjet_jets {suffix_pl} {suffix_gl} --dataset-cap {DSCAP}
    """
    return file

def get_slurm_file_text_AK(tag, log_number):
    bindings = "-B /t3home/gkrzmanc/ -B /work/gkrzmanc/"
    partition = "standard"
    account = "t3"
    d = "jobs/logs/{}".format(tag)
    err = d + "_{}_CPUerr.txt".format(log_number)
    log = d + "_{}_CPUlog.txt".format(log_number)
    file = f"""#!/bin/bash
#SBATCH --partition={partition}           # Specify the partition
#SBATCH --account={account}                  # Specify the account
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=02:00:00               # Set the time limit to 1 hour
#SBATCH --job-name=SVJan  # Name the job
#SBATCH --error={err}         # Redirect stderr to a log file
#SBATCH --output={log}         # Redirect stderr to a log file
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gkrzmanc@student.ethz.ch
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache

nvidia-smi
srun singularity exec {bindings} docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input {args.input} --output {args.input}/batch_eval_2k/{tag}/AK8  --dataset-cap 1500  
srun singularity exec {bindings} docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input {args.input} --output {args.input}/batch_eval_2k/{tag}/AK8_GenJets --jets-object genjets --dataset-cap {DSCAP}
    """
    return file

def get_slurm_file_text(tag, eval_job_name, log_number, aug_suffix = ""):
    bindings = "-B /t3home/gkrzmanc/ -B /work/gkrzmanc/  -B /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc/ "
    partition = "standard"
    account = "t3"
    d = "jobs/logs/{}".format(tag)
    err = d + "_{}_CPUerr.txt".format(log_number)
    log = d + "_{}_CPUlog.txt".format(log_number)
    clust_suffix = ""
    if args.clustering_suffix != "":
        clust_suffix = f" --clustering-suffix {args.clustering_suffix}"
    file = f"""#!/bin/bash
#SBATCH --partition={partition}           # Specify the partition
#SBATCH --account={account}               # Specify the account
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=02:00:00               # Set the time limit to 1 hour
#SBATCH --job-name=SVJ_CPU_{eval_job_name}  # Name the job
#SBATCH --error={err}         # Redirect stderr to a log file
#SBATCH --output={log}         # Redirect stderr to a log file
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gkrzmanc@student.ethz.ch
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi
srun singularity exec {bindings} docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input {args.input} --output {args.input}/batch_eval_2k/{tag}/{eval_job_name}{args.clustering_suffix} --eval-dir train/{eval_job_name} --jets-object model_jets --dataset-cap {DSCAP} {aug_suffix} {clust_suffix}
    """
    return file

runs, run_config = get_eval_run_names(args.tag)
print("RUNS:", runs)


if args.submit_AK8:
   # Submit also ak and ak8
    if not os.path.exists("jobs/slurm_files"):
        os.makedirs("jobs/slurm_files")
    if not os.path.exists("jobs/logs"):
        os.makedirs("jobs/logs")
    log_number = get_log_number(args.tag)
    slurm_file_text = get_slurm_file_text_AK(args.tag, log_number)
    # write the file to jobs/slurm_files
    with open("jobs/slurm_files/evalCPU_{}_{}.slurm".format(args.tag, log_number), "w") as f:
        f.write(slurm_file_text)
        print("Wrote file to jobs/slurm_files/evalCPU_{}_{}.slurm".format(args.tag, log_number))
    if not args.no_submit:
        os.system("sbatch jobs/slurm_files/evalCPU_{}_{}.slurm".format(args.tag, log_number))
    print("---- Submitted AK8 run -----")
    sys.exit(0)


if args.submit_AKX:
    # Submit also AKX
    if not os.path.exists("jobs/slurm_files"):
        os.makedirs("jobs/slurm_files")
    if not os.path.exists("jobs/logs"):
        os.makedirs("jobs/logs")
    log_number = get_log_number(args.tag)
    slurm_file_text = get_slurm_file_text_AKX(args.tag, log_number)
    # write the file to jobs/slurm_files
    with open("jobs/slurm_files/evalCPU_{}_{}.slurm".format(args.tag, log_number), "w") as f:
        f.write(slurm_file_text)
        print("Wrote file to jobs/slurm_files/evalCPU_{}_{}.slurm".format(args.tag, log_number))
    if not args.no_submit:
        os.system("sbatch jobs/slurm_files/evalCPU_{}_{}.slurm".format(args.tag, log_number))
    print("---- Submitted AKX run -----")
    sys.exit(0)

for i, run in enumerate(runs):
    #if get_run_by_name(run).state != "finished":
    #    print("Run not finished (failed or still in progress) - skipping", run)
    #    continue
    aug_soft_p = get_run_by_name(run).config.get("augment_soft_particles", False)
    if aug_soft_p:
        aug_suffix = "-aug-soft"
    else:
        aug_suffix = ""
    if not os.path.exists("jobs/slurm_files"):
        os.makedirs("jobs/slurm_files")
    if not os.path.exists("jobs/logs"):
        os.makedirs("jobs/logs")
    log_number = get_log_number(args.tag)
    slurm_file_text = get_slurm_file_text(args.tag, run, log_number, aug_suffix)
    rel_path_save = f"{args.input}/batch_eval_2k/{args.tag}/{run}{args.clustering_suffix}"
    rel_path_save = get_path(rel_path_save, "results")
    if not os.path.exists(rel_path_save):
        os.makedirs(rel_path_save)
    #if evaluated(rel_path_save):
    if os.path.exists(os.path.join(rel_path_save, "count_matched_quarks", "eval_done.txt")):
        print("Skipping", run, "because this file exists:", os.path.join(rel_path_save, "count_matched_quarks", "eval_done.txt"))
        continue
    else:
        print("Evaluating", run)
    # save run config here
    with open(f"{rel_path_save}/run_config.pkl", "wb") as f:
        pickle.dump(run_config[i], f)
    # write the file to jobs/slurm_files
    with open("jobs/slurm_files/evalCPU_{}_{}.slurm".format(args.tag, log_number), "w") as f:
        f.write(slurm_file_text)
        print("Wrote file to jobs/slurm_files/evalCPU_{}_{}.slurm".format(args.tag, log_number))
    if not args.no_submit:
        os.system("sbatch jobs/slurm_files/evalCPU_{}_{}.slurm".format(args.tag, log_number))
