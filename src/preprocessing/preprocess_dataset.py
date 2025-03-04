import torch
import os.path as osp
import os
import sys
from src.dataset.dataset import SimpleIterDataset
from src.utils.utils import to_filelist
from pathlib import Path
import pickle
from src.utils.paths import get_path
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--dataset-cap", type=int, default=-1)
parser.add_argument("--v2", action="store_true") # V2 means that the dataset also stores parton-level and genParticles

args = parser.parse_args()
path = get_path(args.input, "data")

def remove_from_list(lst):
    out = []
    for item in lst:
        if item in ["hgcal", "data.txt", "test_file.root"]:
            continue
        out.append(item)
    return out

def preprocess_dataset(path, output_path, config_file, dataset_cap):
    datasets = os.listdir(path)
    datasets = [os.path.join(path, x) for x in datasets]
    datasets = datasets
    class Args:
        def __init__(self):
            self.data_train = datasets
            self.data_val = datasets
            #self.data_train = files_train
            self.data_config = config_file
            self.extra_selection = None
            self.train_val_split = 1.0
            self.data_fraction = 1
            self.file_fraction = 1
            self.fetch_by_files = False
            self.fetch_step = 1
            self.steps_per_epoch = None
            self.in_memory = False
            self.local_rank = None
            self.copy_inputs = False
            self.no_remake_weights = False
            self.batch_size = 10
            self.num_workers = 0
            self.demo = False
            self.laplace = False
            self.diffs = False
            self.class_edges = False

    args = Args()
    train_range = (0, args.train_val_split)
    train_file_dict, train_files = to_filelist(args, 'train')
    train_data = SimpleIterDataset(train_file_dict, args.data_config, for_training=True,
                                   extra_selection=args.extra_selection,
                                   remake_weights=True,
                                   load_range_and_fraction=(train_range, args.data_fraction),
                                   file_fraction=args.file_fraction,
                                   fetch_by_files=args.fetch_by_files,
                                   fetch_step=args.fetch_step,
                                   infinity_mode=False,
                                   in_memory=args.in_memory,
                                   async_load=False,
                                   name='train', jets=True)
    iterator = iter(train_data)
    from time import time
    t0 = time()
    data = []
    count = 0
    while True:
        try:
            i = next(iterator)
            data.append(i)
            count += 1
            if dataset_cap > 0 and count >= dataset_cap:
                break
        except StopIteration:
            break
    t1 = time()
    print("Took", t1-t0, "s -", path)
    from src.dataset.functions_data import concat_events
    events = concat_events(data) # TODO: This can be done in a nicer way, using less memory (?)
    result = events.serialize()
    dir_name = path.split("/")[-1]
    save_to_dir = os.path.join(output_path, dir_name)
    Path(save_to_dir).mkdir(parents=True, exist_ok=True)
    for key in result[0]:
        with open(osp.join(save_to_dir, key + ".pkl"), "wb") as f:
            #pickle.dump(result[0][key], f) #save with torch for mmap
            #torch.save(result[0][key], f)
            np.save(f, result[0][key].numpy())
    with open(osp.join(save_to_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(result[1], f)
    print("Saved to", save_to_dir)
    print("Finished", dir_name)
    '''
    from src.dataset.functions_data import EventCollection, EventJets, Event
    from src.dataset.dataset import EventDataset
    t2 = time()
    data1 = []
    for event in EventDataset(result[0], result[1]):
        data1.append(event)
    t3 = time()
    print("Took", t3-t2, "s")
    print("Done")
    '''

output = get_path(args.output, "preprocessed_data")
for dir in os.listdir(path):
    if args.overwrite or not os.path.exists(os.path.join(output, dir)):
        config = get_path('config_files/config_jets.yaml', 'code')
        if args.v2:
            config = get_path('config_files/config_jets_1.yaml', 'code')
        preprocess_dataset(os.path.join(path, dir), output, config_file=config, dataset_cap=args.dataset_cap)
    else:
        print("Skipping", dir + ", already exists")
        # flush
        sys.stdout.flush()