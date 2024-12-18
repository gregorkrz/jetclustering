import os
from src.dataset.dataset import EventDataset
from src.utils.paths import get_path
import argparse

# This script attempts to open dataset files and prints the number of events in each one.

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
#parser.add_argument("--output", type=str)
args = parser.parse_args()
path = get_path(args.input, "preprocessed_data")

for dataset in sorted(os.listdir(path)):
    ds = EventDataset.from_directory(os.path.join(path, dataset))
    print(dataset + ":" , len(ds), "events")

print("------------------")
