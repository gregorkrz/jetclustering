import argparse

parser = argparse.ArgumentParser()

######### Data-related arguments #########

parser.add_argument("-c", "--data-config", type=str, help="data config YAML file", default="config_files/config_jets.yaml")

parser.add_argument(
    "-train",
    "--data-train",
    nargs="*",
    default=[],
    help="training files; supported syntax:"
    " (a) plain list, `--data-train /path/to/a/* /path/to/b/*`;"
    " (b) (named) groups [Recommended], `--data-train a:/path/to/a/* b:/path/to/b/*`,"
    " the file splitting (for each dataloader worker) will be performed per group,"
    " and then mixed together, to ensure a uniform mixing from all groups for each worker.",
)
parser.add_argument(
    "-val",
    "--data-val",
    nargs="*",
    help="validation files",
)
parser.add_argument(
    "-tag",
    "--tag",
    type=str,
    required=False
)
parser.add_argument(
    "-ckpt-step",
    "--ckpt-step",
    type=int,
    required=False,
    default=0
) # to make it easier to find the actual number of steps

parser.add_argument(
    "-load-from-run",
    "--load-from-run",
    required=False,
    default="",
    type=str,
    help="WandB run name from which to pull the training settings"
)

parser.add_argument("--train-dataset-size", type=int, default=None, help="number of events to use from the training dataset")
parser.add_argument("--val-dataset-size", type=int, default=None, help="number of events to use from the validation dataset")
parser.add_argument(
    "-test",
    "--data-test",
    nargs="*",
    default=[],
    help="testing files; supported syntax:"
    " (a) plain list, `--data-test /path/to/a/* /path/to/b/*`;"
    " (b) keyword-based, `--data-test a:/path/to/a/* b:/path/to/b/*`, will produce output_a, output_b;"
    " (c) split output per N input files, `--data-test a%10:/path/to/a/*`, will split per 10 input files",
)

######### Model and training-related arguments #########

parser.add_argument(
    "-net",
    "--network-config",
    type=str,
    help="network architecture configuration file; the path must be relative to the current dir",
)

parser.add_argument(
    "-n-blocks",
    "--num-blocks",
    type=int,
    help="Number of blocks for GATr/LGATr/Transformer",
    required=False,
    default=10
)

##### Transformer-specific arguments #####

parser.add_argument(
    "-internal-dim",
    "--internal-dim",
    type=int,
    help="Internal dim for transformer",
    required=False,
    default=128
)

parser.add_argument(
    "-heads",
    "--n-heads",
    type=int,
    help="N attention heads for transformer",
    required=False,
    default=4
)

##### L-GATr-specific arguments #####

parser.add_argument(
    "-mv-ch",
    "--hidden-mv-channels",
    type=int,
    help="Hidden multivector channels for gatr and l-gatr",
    required=False,
    default=16
)

parser.add_argument(
    "-s-ch",
    "--hidden-s-channels",
    type=int,
    help="Hidden scalar channels for GATr and L-GATr",
    required=False,
    default=64
)


parser.add_argument(
    "--load-model-weights",
    type=str,
    default=None,
    help="initialize model with pre-trained weights",
)

parser.add_argument(
    "--run-name",
    type=str,
    help="The name of the run. The wandb name and the folder it gets saved to will be this name + timestamp.",
)

parser.add_argument(
    "--prefix",
    type=str,
    default="",
    help="Path to the results folder, if empty, it will be set to the results folder in the current environment.",
)

parser.add_argument(
    "--debug",
    action="store_true",
    default=False,
    help="quickly test the setup by running over only a small number of events - use for debugging",
)

parser.add_argument(
    "--wandb-projectname", type=str, help="project where the run is stored inside wandb", default="svj_clustering"
)

parser.add_argument("--batch-size", "-bs", type=int, default=128, help="batch size")
parser.add_argument("--num-epochs", type=int, default=20, help="number of epochs")
parser.add_argument("--num-steps", type=int, default=-1, help="Number of steps. If set to -1, it will be ignored and only num_epochs will be considered. Otherwise, training will stop after the reached number of steps.")

parser.add_argument(
    "--gpus",
    type=str,
    default="0",
    help='device for the training/testing; to use CPU, set to empty string (""); to use multiple gpu, set it as a comma separated list, e.g., `1,2,3,4`',
)

parser.add_argument(
    "--num-workers",
    type=int,
    default=1,
    help="number of threads to load the dataset; memory consumption and disk access load increases (~linearly) with this numbers",
)


### Loss-related arguments ###

parser.add_argument(
    "--loss",
    type=str,
    default="oc",
    choices=["oc", "quark_distance"],
    help="Loss function to use (oc is object condensation, quark_distance aims to cluster things around the corresponding dark quark)"
)

parser.add_argument("--gt-radius", type=float, default=0.8, help="GT radius R - within the radius of a dark quark, GT points to the dark quark, out of the radius it's noise")

parser.add_argument("--attr-loss-weight", type=float, default=1.0, help="weight for the attractive loss")
parser.add_argument("--repul-loss-weight", type=float, default=1.0, help="weight for the repulsive loss")
parser.add_argument("--coord-loss-weight", type=float, default=0.0, help="weight for the coordinate loss")
parser.add_argument(
    "--beta-type",
    type=str,
    default="default",
    choices=["default", "pt", "pt+bc"],
    help="How to predict betas",
)

parser.add_argument(
    "--lorentz-norm",
    help="Whether the norm in clustering space should be the Lorentz one (otherwise it's usual Euclidean 2-norm)",
    action="store_true",
    default=False,
)

parser.add_argument(
    "--scalars-oc",
    help="For L-GATr, use scalar virtual coordinates in the OC loss",
    action="store_true",
    default=False,
)

parser.add_argument(
    "--spatial-part-only",
    help="For L-GATr: if turned on, the spatial part is only going to be used for the loss.",
    action="store_true",
    default=False,
)


# defaults: --min-cluster-size 11 --min-samples 18 --epsilon 0.48

parser.add_argument(
    "--min-cluster-size",
    help="parameter of the HDBSCAN clustering",
    type=int,
    default=11
)

parser.add_argument(
    "--min-samples",
    help="parameter of the HDBSCAN clustering",
    type=int,
    default=18
)
parser.add_argument(
    "--epsilon",
    help="parameter of the HDBSCAN clustering",
    type=float,
    default=0.48
)


parser.add_argument(
    "-embed-as-vectors",
    "--embed-as-vectors",
    action="store_true",
    default=False,
    help="Whether to embed the input p_xyz as vectors (translations) or points",
)

#### Optimizer and LR-related arguments ####

parser.add_argument(
    "--optimizer",
    type=str,
    default="ranger",
    choices=["adam", "adamW", "radam", "ranger"],
    help="optimizer for the training",
)
parser.add_argument(
    "--optimizer-option",
    nargs=2,
    action="append",
    default=[],
    help="options to pass to the optimizer class constructor, e.g., `--optimizer-option weight_decay 1e-4`",
)
parser.add_argument(
    "--lr-scheduler",
    type=str,
    default="flat+decay",
    choices=[
        "none",
        "steps",
        "flat+decay",
        "flat+linear",
        "flat+cos",
        "one-cycle",
        "reduceplateau",
    ],
    help="learning rate scheduler",
)
parser.add_argument("--start-lr", type=float, default=5e-3, help="start learning rate")
parser.add_argument("--validation-steps", type=float, default=1000, help="Run eval on validation set every x steps")

parser.add_argument(
    "--backend",
    type=str,
    choices=["gloo", "nccl", "mpi"],
    default=None,
    help="backend for distributed training",
)
parser.add_argument(
    "--log",
    type=str,
    default="",
    help="path to the log file; `{auto}` can be used as part of the path to auto-generate a name, based on the timestamp and network configuration",
)
parser.add_argument(
    "--use-amp",
    action="store_true",
    default=False,
    help="use mixed precision training (fp16)",
)
parser.add_argument(
    "-obj-score",
    "--train-objectness-score",
    action="store_true",
    help="Whether to train the objectness classifier next to the usual clustering loss",
)

parser.add_argument(
    "-obj-score-weights",
    "--load-objectness-score-weights",
    type=str,
    help="Ckpt file for the objectness score model",
    default="",
    required=False
)

