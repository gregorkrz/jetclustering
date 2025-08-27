import os
import wandb
from src.utils.paths import get_path

api = wandb.Api()

def get_run_by_name(name):
    runs = api.runs(
        path="fcc_ml/svj_clustering",
        filters={"display_name": {"$eq": name.strip()}}
    )
    runs = api.runs(
        path="fcc_ml/svj_clustering",
        filters={"display_name": {"$eq": name.strip()}}
    )

    if runs.length != 1:
        return None
    return runs[0]

def get_steps_from_file(fname):
    return int(fname.split("/")[-1].split("_")[1])

def get_run_initial_steps(run):
    if not run.config["load_model_weights"]:
        return 0
    else:
        run_name_1 = run.config["load_model_weights"].split("/")[-2]
        run_1 = get_run_by_name(run_name_1)
        if run_1 is None: raise Exception("Run doesn't exist: " + run_name_1)
        return get_run_initial_steps(run_1) + get_steps_from_file(run.config["load_model_weights"])

def extract_relative_path(run_path):
    # just return everything after train/.. - run_path looks like /a/b/c/d/train/e/f
    return get_path("train/" + run_path.split("train/")[-1], type="results", fallback=True)
    #return "train/" + run_path.split("train/")[-1]


def get_run_step_direct(run_path, step):
    # get the step of the run directly
    p = extract_relative_path(run_path)
    print("Run-path:", p)
    lst = os.listdir(p)
    lst = [x for x in lst if x.endswith(".ckpt")] # files are of format step_x_epoch_y.ckpt
    steps = [int(x.split("_")[1]) for x in lst]
    if step not in steps:
        print("Available steps:", steps)
        raise Exception("Step not found in run")
    full_path = os.path.join(p, [x for x in lst if int(x.split("_")[1]) == step][0])
    # return everything after "train/"
    return "train/" + full_path.split("train/")[-1]


def get_run_step_ckpt(run, step, steps_from_zero):
    if not run.config["load_model_weights"] or steps_from_zero:
        return get_run_step_direct(run.config["run_path"], step), run
    else:
        run_name_1 = run.config["load_model_weights"].split("/")[-2]
        run_1 = get_run_by_name(run_name_1)
        if run_1 is None: raise Exception("Run doesn't exist: " + run_name_1)
        steps = get_run_initial_steps(run)
        if step > steps:
            print("Step", step, "is in run", run.name)
            return get_run_step_direct(run_1.config["run_path"], step - steps), run_1
        else:
            return get_run_step_ckpt(run_1, step)

args_to_update = ["validation_steps", "start_lr", "lr_scheduler", "optimizer", "embed_as_vectors", "epsilon",
                  "min_samples", "min_cluster_size", "spatial_part_only", "scalars_oc", "lorentz_norm", "beta_type",
                  "coord_loss_weight", "repul_loss_weight", "attr_loss_weight", "gt_radius", "loss", "num_steps",
                  "num_epochs", "hidden_s_channels", "hidden_mv_channels", "n_heads", "internal_dim",
                  "num_blocks", "network_config", "data_config", "no_pid"]

def update_args(args, run):
    for arg in args_to_update:
        if arg in ["min_samples", "min_cluster_size", "epsilon"]:
            print("Skipping setting clustering args")
            continue
        if arg not in run.config:
            print("Skipping setting", arg)
            continue
        print("Setting", arg, run.config[arg])
        setattr(args, arg, run.config[arg])
    print("Loaded args from run", run.name)
    args.parent_run = run.name
    return args

