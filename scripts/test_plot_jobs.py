# Eval a given a training run name at the given steps, taking into account the chaning of the training runs
import wandb
import argparse
import os

from src.utils.paths import get_path

from src.utils.wandb_utils import get_run_initial_steps, get_run_step_direct, get_run_step_ckpt, get_steps_from_file, get_run_by_name

parser = argparse.ArgumentParser()
parser.add_argument("--tag", "-tag", type=str, required=False, default="") # 'Vega' or 'T3'
parser.add_argument("--input", "-input", type=str, required=False, default="scouting_PFNano_signals2/SVJ_hadronic_std")
parser.add_argument("--no-submit", "-ns", action="store_true") # do not submit the slurm job

args = parser.parse_args()

api = wandb.Api()

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
    file = f"""#!/bin/bash
#SBATCH --partition={partition}           # Specify the partition
#SBATCH --account={account}                  # Specify the account
#SBATCH --mem=10000                   # Request 10GB of memory
#SBATCH --time=02:00:00               # Set the time limit to 1 hour
#SBATCH --job-name=SVJan  # Name the job
#SBATCH --error={err}         # Redirect stderr to a log file
#SBATCH --output={log}         # Redirect stderr to a log file
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi
srun singularity exec {bindings} docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input {args.input} --output {args.input}/batch_eval/{tag}/AKX --jets-object fastjet_jets
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
#SBATCH --mem=10000                   # Request 10GB of memory
#SBATCH --time=02:00:00               # Set the time limit to 1 hour
#SBATCH --job-name=SVJan  # Name the job
#SBATCH --error={err}         # Redirect stderr to a log file
#SBATCH --output={log}         # Redirect stderr to a log file
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi
srun singularity exec {bindings} docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input {args.input} --output {args.input}/batch_eval/{tag}/AK8 
srun singularity exec {bindings} docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input {args.input} --output {args.input}/batch_eval/{tag}/AK8_GenJets --jets-object genjets
    """
    return file

def get_slurm_file_text(tag, eval_job_name, log_number):
    bindings = "-B /t3home/gkrzmanc/ -B /work/gkrzmanc/  -B /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc/ "
    partition = "standard"
    account = "t3"
    d = "jobs/logs/{}".format(tag)
    err = d + "_{}_CPUerr.txt".format(log_number)
    log = d + "_{}_CPUlog.txt".format(log_number)
    file = f"""#!/bin/bash
#SBATCH --partition={partition}           # Specify the partition
#SBATCH --account={account}                  # Specify the account
#SBATCH --mem=10000                   # Request 10GB of memory
#SBATCH --time=02:00:00               # Set the time limit to 1 hour
#SBATCH --job-name=SVJan  # Name the job
#SBATCH --error={err}         # Redirect stderr to a log file
#SBATCH --output={log}         # Redirect stderr to a log file
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi
srun singularity exec {bindings} docker://gkrz/lgatr:v3 python -m scripts.analysis.count_matched_quarks --input {args.input} --output {args.input}/batch_eval/{tag}/{eval_job_name} --eval-dir train/{eval_job_name} --jets-object model_jets
    """
    return file

runs, run_config = get_eval_run_names(args.tag)
print("RUNS:", runs)

'''# Submit also ak and ak8
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

print("---- Submitted AK8 run -----")'''

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

import pickle

for i, run in enumerate(runs):
    if not os.path.exists("jobs/slurm_files"):
        os.makedirs("jobs/slurm_files")
    if not os.path.exists("jobs/logs"):
        os.makedirs("jobs/logs")
    log_number = get_log_number(args.tag)
    slurm_file_text = get_slurm_file_text(args.tag, run, log_number)
    rel_path_save = f"{args.input}/batch_eval/{args.tag}/{run}"
    rel_path_save = get_path(rel_path_save, "results")
    if not os.path.exists(rel_path_save):
        os.makedirs(rel_path_save)
    else:
        print("Skipping", run)
        continue
    # save run config here
    with open(f"{rel_path_save}/run_config.pkl", "wb") as f:
        pickle.dump(run_config[i], f)
    # write the file to jobs/slurm_files
    with open("jobs/slurm_files/evalCPU_{}_{}.slurm".format(args.tag, log_number), "w") as f:
        f.write(slurm_file_text)
        print("Wrote file to jobs/slurm_files/evalCPU_{}_{}.slurm".format(args.tag, log_number))
    if not args.no_submit:
        os.system("sbatch jobs/slurm_files/evalCPU_{}_{}.slurm".format(args.tag, log_number))
