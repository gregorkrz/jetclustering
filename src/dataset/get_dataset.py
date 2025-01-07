import os
from src.dataset.dataset import SimpleIterDataset, EventDataset
from src.utils.utils import to_filelist
from src.utils.paths import get_path


# To be used for simple analysis scripts, not for the full training!
def get_iter(path, full_dataloader=False, model_clusters_dir=None):
    if full_dataloader:
        datasets = os.listdir(path)
        datasets = [os.path.join(path, x) for x in datasets]
        class Args:
            def __init__(self):
                self.data_train = datasets
                self.data_val = datasets
                #self.data_train = files_train
                self.data_config = get_path('config_files/config_jets.yaml', "code")
                self.extra_selection = None
                self.train_val_split = 1
                self.data_fraction = 1
                self.file_fraction = 1
                self.fetch_by_files = False
                self.fetch_step = 0.1
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
    else:
        iterator = iter(EventDataset.from_directory(path, model_clusters_dir=model_clusters_dir))
    return iterator
