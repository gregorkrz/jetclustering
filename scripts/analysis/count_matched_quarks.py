import os
from tqdm import tqdm
import argparse

from src.dataset.get_dataset import get_iter
from src.utils.paths import get_path



# This script attempts to open dataset files and prints the number of events in each one.

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True)
args = parser.parse_args()
path = get_path(args.input, "preprocessed_data")
dataset = get_iter(path)


for data in tqdm(dataset):
    jet = [data.fatjets.eta, data.fatjets.phi]
    quark = [data.matrix_element_gen_particles.eta, data.matrix_element_gen_particles.phi]
    pass

