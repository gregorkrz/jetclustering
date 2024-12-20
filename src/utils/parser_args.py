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
    required=True,
    help="validation files",
)
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

parser.add_argument(
    "--predict",
    action="store_true",
    default=False,
    help="run prediction instead of training",
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