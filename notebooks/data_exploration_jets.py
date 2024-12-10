import torch
import sys
import os.path as osp
import os
import sys
import numpy as np
sys.path.append("/eos/home-g/gkrzmanc/jetclustering/code")
from src.dataset.dataset import SimpleIterDataset
from src.utils.utils import to_filelist
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from torch_scatter import scatter_sum
import matplotlib.pyplot as plt
import pickle
import numpy as np
import mplhep as hep
hep.style.use("CMS")
import matplotlib
matplotlib.rc('font', size=13)


def remove_from_list(lst):
    out = []
    for item in lst:
        if item in ["hgcal", "data.txt", "test_file.root"]:
            continue
        out.append(item)
    return out

path = "/eos/user/g/gkrzmanc/jetclustering/data/SVJ_std_UL2018_scouting_test_large/SVJ_mMed-700GeV_mDark-20GeV_rinv-0.7_alpha-peak"

datasets = os.listdir(path)
datasets = [os.path.join(path, x) for x in datasets]
datasets = datasets[:5]

class Args:
    def __init__(self):
        self.data_train = datasets
        self.data_val = datasets
        #self.data_train = files_train
        self.data_config = '/eos/home-g/gkrzmanc/jetclustering/code/config_files/config_jets.yaml'
        self.extra_selection = None
        self.train_val_split = 0.8
        self.data_fraction = 1
        self.file_fraction = 1
        self.fetch_by_files = False
        self.fetch_step = 0.2
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
for i in range(100):
    data.append(next(iterator))
t1 = time()
print("Took", t1-t0, "s")
from src.dataset.functions_data import concat_events
events = concat_events(data) # TODO: this can be done in a nicer way, using less memory (?)
result = events.serialize()
save_to_dir = "/eos/user/g/gkrzmanc/jetclustering/data/torch_SVJ_std_UL2018_scouting_test_large/SVJ_mMed-700GeV_mDark-20GeV_rinv-0.7_alpha-peak"
Path(save_to_dir).mkdir(parents=True, exist_ok=True)
for key in result[0]:
    with open(osp.join(save_to_dir, key + ".pkl"), "wb") as f:
        #pickle.dump(result[0][key], f) #save with torch for mmap
        torch.save(result[0][key], f)
with open(osp.join(save_to_dir, "metadata.pkl"), "wb") as f:
    pickle.dump(result[1], f)

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