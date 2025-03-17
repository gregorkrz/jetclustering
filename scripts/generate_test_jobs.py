# Eval a given a training run name at the given steps, taking into account the chaning of the training runs
import wandb
import argparse
import os

from src.utils.wandb_utils import get_run_initial_steps, get_run_step_direct, get_run_step_ckpt, get_steps_from_file, get_run_by_name

parser = argparse.ArgumentParser()
parser.add_argument("--train-run-name", "-run", type=str, required=True)
parser.add_argument("--steps", "-step", type=int, required=True)
parser.add_argument("--template", "-template", type=str, required=True) # 'Vega' or 'T3'
parser.add_argument("--tag", "-tag", type=str, required=False, default="") # 'Vega' or 'T3'
parser.add_argument("--no-submit", "-ns", action="store_true") # do not submit the slurm job
parser.add_argument("--os-weights", "-os", default=None, type=str)  # train/scatter_mean_Obj_Score_LGATr_8_16_64_2025_02_07_16_31_26/OS_step_47000_epoch_70.ckpt # objectness score weights
parser.add_argument("--global-features-obj-score", "-glob-f", action="store_true") # use global features for objectness score (setting of the OS run, not of the clustering run)
parser.add_argument("--parton-level", "-pl", action="store_true") # use parton level
parser.add_argument("--gen-level", "-gl", action="store_true") # use gen level
parser.add_argument("--custom-test-files", type=str, default="")

# -os train/scatter_mean_Obj_Score_LGATr_8_16_64_2025_02_07_16_31_26/OS_step_47000_epoch_70.ckpt
args = parser.parse_args()

"""
python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_0.8_2025_02_28_12_42_59 -step 40000 -template t3 -tag no_pid_eval_1 -gl --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"
python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_2.0_2025_02_28_12_48_58 -step 40000 -template t3 -tag no_pid_eval_1 -gl --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"
python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_0.8_700_07_2025_02_28_13_01_59 -step 40000 -template t3 -tag no_pid_eval_1 -gl --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"
python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_0.8_AllData_2025_02_28_13_42_59 -step 40000 -template t3 -tag no_pid_eval_1 -gl --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"

python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_0.8_2025_02_28_12_42_59 -step 40000 -template t3 -tag no_pid_eval_1 -pl --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"
python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_2.0_2025_02_28_12_48_58 -step 40000 -template t3 -tag no_pid_eval_1 -pl --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"
python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_0.8_700_07_2025_02_28_13_01_59 -step 40000 -template t3 -tag no_pid_eval_1 -pl --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"
python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_0.8_AllData_2025_02_28_13_42_59 -step 40000 -template t3 -tag no_pid_eval_1 -pl --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"


python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_0.8_2025_02_28_12_42_59 -step 40000 -template t3 -tag no_pid_eval_1 --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"
python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_2.0_2025_02_28_12_48_58 -step 40000 -template t3 -tag no_pid_eval_1 --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"
python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_0.8_700_07_2025_02_28_13_01_59 -step 40000 -template t3 -tag no_pid_eval_1 --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"
python -m scripts.generate_test_jobs -run LGATr_training_NoPID_10_16_64_0.8_AllData_2025_02_28_13_42_59 -step 40000 -template t3 -tag no_pid_eval_1 --custom-test-files "Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000 Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-700_mDark-20_rinv-0.7_alpha-peak_13TeV-pythia8_n-1000  Feb26_2025_E1000_N500_folders/PFNano_s-channel_mMed-1200_mDark-20_rinv-0.3_alpha-peak_13TeV-pythia8_n-1000"

"""




api = wandb.Api()
import time

# Dummy API call, for some reason it doesn't work without this?????
'''
train_run = get_run_by_name(args.train_run_name)
print(get_run_initial_steps(train_run))
print(get_run_initial_steps(get_run_by_name("lgatr_CONT_ds_cap_5000_2025_01_21_19_46_13")))
print(get_run_step_ckpt(get_run_by_name("lgatr_CONT_ds_cap_5000_2025_01_21_19_46_13"), 20000))
print(get_run_step_ckpt(get_run_by_name("lgatr_CONT_ds_cap_5000_2025_01_21_19_46_13"), 21000))
print(get_run_step_ckpt(get_run_by_name("lgatr_CONT_ds_cap_5000_2025_01_21_19_46_13"), 15000))
'''

ckpt_file, train_run = get_run_step_ckpt(get_run_by_name(args.train_run_name), args.steps)

test_files = "scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-700_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-900_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1500_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-800_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1000_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1000_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-3000_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-800_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-700_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-900_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-3000_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-800_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1500_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1000_mDark-20_rinv-0.5 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-1500_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-3000_mDark-20_rinv-0.3 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-900_mDark-20_rinv-0.7 scouting_PFNano_signals2/SVJ_hadronic_std/s-channel_mMed-700_mDark-20_rinv-0.7"
if args.custom_test_files:
    test_files = args.custom_test_files
    print("Using custom test files:", test_files)

def get_log_number(tag):
    numbers = set()
    for file in os.listdir("jobs/logs"):
        if tag in file:
            numbers.add(int(file.split("_")[-2]))
    if len(numbers) == 0:
        return 0
    return max(list(numbers)) + 1


def get_slurm_file_text(template, run_name, tag, ckpt_file, log_number):
    bindings = "-B /t3home/gkrzmanc/ -B /work/gkrzmanc/ -B /pnfs/psi.ch/cms/trivcat/store/user/gkrzmanc/ "
    partition = "gpu"
    account = "gpu_gres"
    if template.lower().strip() == "vega":
        bindings = " -B /ceph/hpc/home/krzmancg "
        account = "s25t01-01-users"
    tag_suffix = ""
    if tag:
        tag_suffix = " --tag " + tag
    d = "jobs/logs/{}".format(tag)
    err = d + "_{}_err.txt".format(log_number)
    log = d + "_{}_log.txt".format(log_number)
    obj_score_suffix = ""
    glob_features_obj_score_suffix = ""
    eval_suffix = ""
    if args.parton_level:
        eval_suffix = " --parton-level "
    elif args.gen_level:
        eval_suffix = " --gen-level "
    if args.global_features_obj_score:
        glob_features_obj_score_suffix = " --global-features-obj-score"
    if args.os_weights:
        obj_score_suffix = f" --train-objectness-score --load-objectness-score-weights {args.os_weights}"
    file = f"""#!/bin/bash
#SBATCH --partition={partition}           # Specify the partition
#SBATCH --account={account}               # Specify the account
#SBATCH --mem=25000                   # Request 10GB of memory
#SBATCH --time=02:00:00               # Set the time limit to 1 hour
#SBATCH --job-name=SVJeval  # Name the job
#SBATCH --error={err}       # Redirect stderr to a log file
#SBATCH --output={log}      # Redirect stderr to a log file
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=gkrzmanc@student.ethz.ch
source env.sh
export APPTAINER_TMPDIR=/work/gkrzmanc/singularity_tmp
export APPTAINER_CACHEDIR=/work/gkrzmanc/singularity_cache
nvidia-smi
srun singularity exec {bindings} --nv docker://gkrz/lgatr:v3 python -m src.train -test {test_files} --gpus 0 --run-name Eval_{tag} --load-model-weights {ckpt_file} --num-workers 0 {tag_suffix} --load-from-run {run_name} --ckpt-step {args.steps} {obj_score_suffix} {glob_features_obj_score_suffix} {eval_suffix} --epsilon 0.5 --min-samples 2 --min-cluster-size 4
    """
    return file



if not os.path.exists("jobs/slurm_files"):
    os.makedirs("jobs/slurm_files")
if not os.path.exists("jobs/logs"):
    os.makedirs("jobs/logs")

log_number = get_log_number(args.tag)
slurm_file_text = get_slurm_file_text(args.template, args.train_run_name, args.tag, ckpt_file, log_number)
# write the file to jobs/slurm_files
with open("jobs/slurm_files/{}_{}.slurm".format(args.tag, log_number), "w") as f:
    f.write(slurm_file_text)
    print("Wrote file to jobs/slurm_files/{}_{}.slurm".format(args.tag, log_number))

if not args.no_submit:
    os.system("sbatch jobs/slurm_files/{}_{}.slurm".format(args.tag, log_number))
