import os
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import pickle
from src.dataset.get_dataset import get_iter
from src.utils.paths import get_path
from pathlib import Path

# This script attempts to open dataset files and prints the number of events in each one.

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
parser.add_argument("--dataset-cap", type=int, default=-1)
parser.add_argument("--output", type=str, default="")
parser.add_argument("--plot-only", action="store_true")

args = parser.parse_args()
path = get_path(args.input, "preprocessed_data")
if args.output == "":
    args.output = args.input
output_path = os.path.join(get_path(args.output, "results"), "analysis_name")
Path(output_path).mkdir(parents=True, exist_ok=True)

if not args.plot_only:
    pass
    #  Do some computations here
    # pickle.dump(result, open(os.path.join(output_path, "result.pkl"), "wb"))
#if args.plot_only:
#    result = pickle.load(open(os.path.join(output_path, "result.pkl"), "rb"))

import matplotlib.pyplot as plt
# Do some plotting here
