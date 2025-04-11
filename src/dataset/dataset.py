import os
import copy
import json
import numpy as np
import awkward as ak
import torch.utils.data
import time
import pickle
from collections import OrderedDict
from functools import partial
from concurrent.futures.thread import ThreadPoolExecutor
from src.logger.logger import _logger, warn_once
from src.data.tools import _pad, _repeat_pad, _clip, _pad_vector
from src.data.fileio import _read_files
from src.data.config import DataConfig, _md5
from src.data.preprocess import (
    _apply_selection,
    _build_new_variables,
    _build_weights,
    AutoStandardizer,
    WeightMaker,
)
from src.dataset.functions_data import to_tensor
from src.layers.object_cond import calc_eta_phi
from torch_scatter import scatter_sum
from src.dataset.functions_graph import (create_graph, create_jets_outputs,
                                         create_jets_outputs_new, create_jets_outputs_Delphes)
from src.dataset.functions_data import Event, EventCollection, EventJets
import fastjet
from src.utils.utils import CPU_Unpickler
from src.dataset.functions_data import EventPFCands, concat_event_collection

def get_pseudojets_fastjet(pfcands):
    pseudojets = []
    for i in range(len(pfcands)):
        pseudojets.append(fastjet.PseudoJet(pfcands.pxyz[i, 0].item(), pfcands.pxyz[i, 1].item(), pfcands.pxyz[i, 2].item(), pfcands.E[i].item()))
    return pseudojets

def _finalize_inputs(table, data_config):
    # transformation
    output = {}
    # transformation
    for k, params in data_config.preprocess_params.items():
        if data_config._auto_standardization and params["center"] == "auto":
            raise ValueError("No valid standardization params for %s" % k)
        # if params["center"] is not None:
        #     table[k] = (table[k] - params["center"]) * params["scale"]
        if params["length"] is not None:
            # if k == "hit_genlink":
            #    pad_fn = partial(_pad_vector, value=-1)
            #    table[k] = pad_fn(table[k])
            # else:
            pad_fn = partial(_pad, value=0)
            table[k] = pad_fn(table[k], params["length"])

    # stack variables for each input group
    for k, names in data_config.input_dicts.items():
        if (
            len(names) == 1
            and data_config.preprocess_params[names[0]]["length"] is None
        ):
            output["_" + k] = ak.to_numpy(ak.values_astype(table[names[0]], "float32"))
        else:
            output["_" + k] = ak.to_numpy(
                np.stack(
                    [ak.to_numpy(table[n]).astype("float32") for n in names], axis=1
                )
            )
    # copy monitor variables
    for k in data_config.z_variables:
        if k not in output:
            output[k] = ak.to_numpy(table[k])
    return output


def _padlabel(table, _data_config):
    for k in _data_config.label_value:
        pad_fn = partial(_pad, value=0)
        table[k] = pad_fn(table[k], 400)

    return table


def _preprocess(table, data_config, options):
    # apply selection
    table = _apply_selection(
        table,
        data_config.selection
        if options["training"]
        else data_config.test_time_selection,
    )
    if len(table) == 0:
        return []
    # table = _padlabel(table,data_config)
    # define new variables
    table = _build_new_variables(table, data_config.var_funcs)

    # else:
    indices = np.arange(
        len(table[table.fields[0]])
    )  # np.arange(len(table[data_config.label_names[0]]))
    # shuffle
    if options["shuffle"]:
        np.random.shuffle(indices)
    # perform input variable standardization, clipping, padding and stacking
    table = _finalize_inputs(table, data_config)
    return table, indices


def _load_next(data_config, filelist, load_range, options):
    table = _read_files(
        filelist, data_config.load_branches, load_range, treename=data_config.treename
    )
    table, indices = _preprocess(table, data_config, options)
    return table, indices


class _SimpleIter(object):
    r"""_SimpleIter
    Iterator object for ``SimpleIterDataset''.
    """

    def __init__(self, **kwargs):
        # inherit all properties from SimpleIterDataset
        self.__dict__.update(**kwargs)
        self.iter_count = 0  # to raise StopIteration when dataset_cap is reached
        if "dataset_cap" in kwargs and kwargs["dataset_cap"] is not None:
            self.dataset_cap = kwargs["dataset_cap"]
            self._sampler_options["shuffle"] = False
            print("!!! Dataset_cap flag set, disabling shuffling")
        else:
            self.dataset_cap = None

        # executor to read files and run preprocessing asynchronously
        self.executor = ThreadPoolExecutor(max_workers=1) if self._async_load else None

        # init: prefetch holds table and indices for the next fetch
        self.prefetch = None
        self.table = None
        self.indices = []
        self.cursor = 0

        self._seed = None
        worker_info = torch.utils.data.get_worker_info()
        file_dict = self._init_file_dict.copy()
        if worker_info is not None:
            # in a worker process
            self._name += "_worker%d" % worker_info.id
            self._seed = worker_info.seed & 0xFFFFFFFF
            np.random.seed(self._seed)
            # split workload by files
            new_file_dict = {}
            for name, files in file_dict.items():
                new_files = files[worker_info.id :: worker_info.num_workers]
                assert len(new_files) > 0
                new_file_dict[name] = new_files
            file_dict = new_file_dict
        self.worker_file_dict = file_dict
        self.worker_filelist = sum(file_dict.values(), [])
        self.worker_info = worker_info

        self.restart()

    def restart(self):
        print("=== Restarting DataIter %s, seed=%s ===" % (self._name, self._seed))
        # re-shuffle filelist and load range if for training
        filelist = self.worker_filelist.copy()
        if self._sampler_options["shuffle"]:
            np.random.shuffle(filelist)
        if self._file_fraction < 1:
            num_files = int(len(filelist) * self._file_fraction)
            filelist = filelist[:num_files]
        self.filelist = filelist

        if self._init_load_range_and_fraction is None:
            self.load_range = (0, 1)
        else:
            (start_pos, end_pos), load_frac = self._init_load_range_and_fraction
            interval = (end_pos - start_pos) * load_frac
            if self._sampler_options["shuffle"]:
                offset = np.random.uniform(start_pos, end_pos - interval)
                self.load_range = (offset, offset + interval)
            else:
                self.load_range = (start_pos, start_pos + interval)

        _logger.debug(
            "Init iter [%d], will load %d (out of %d*%s=%d) files with load_range=%s:\n%s",
            0 if self.worker_info is None else self.worker_info.id,
            len(self.filelist),
            len(sum(self._init_file_dict.values(), [])),
            self._file_fraction,
            int(len(sum(self._init_file_dict.values(), [])) * self._file_fraction),
            str(self.load_range),
        )
        # '\n'.join(self.filelist[: 3]) + '\n ... ' + self.filelist[-1],)

        _logger.info(
            "Restarted DataIter %s, load_range=%s, file_list:\n%s"
            % (
                self._name,
                str(self.load_range),
                json.dumps(self.worker_file_dict, indent=2),
            )
        )

        # reset file fetching cursor
        self.ipos = 0 if self._fetch_by_files else self.load_range[0]
        # prefetch the first entry asynchronously
        self._try_get_next(init=True)

    def __next__(self):
        # print(self.ipos, self.cursor)
        graph_empty = True
        self.iter_count += 1
        if self.dataset_cap is not None and self.iter_count > self.dataset_cap:
            raise StopIteration
        while graph_empty:
            if len(self.filelist) == 0:
                raise StopIteration
            try:
                i = self.indices[self.cursor]
            except IndexError:
                # case 1: first entry, `self.indices` is still empty
                # case 2: running out of entries, `self.indices` is not empty
                while True:
                    if self._in_memory and len(self.indices) > 0:
                        # only need to re-shuffle the indices, if this is not the first entry
                        if self._sampler_options["shuffle"]:
                            np.random.shuffle(self.indices)
                        break
                    if self.prefetch is None:
                        # reaching the end as prefetch got nothing
                        self.table = None
                        if self._async_load:
                            self.executor.shutdown(wait=False)
                        raise StopIteration
                    # get result from prefetch
                    if self._async_load:
                        self.table, self.indices = self.prefetch.result()
                    else:
                        self.table, self.indices = self.prefetch
                    # try to load the next ones asynchronously
                    self._try_get_next()
                    # check if any entries are fetched (i.e., passing selection) -- if not, do another fetch
                    if len(self.indices) > 0:
                        break
                # reset cursor
                self.cursor = 0
                i = self.indices[self.cursor]
            self.cursor += 1
            data, graph_empty = self.get_data(i)
        return data

    def _try_get_next(self, init=False):
        end_of_list = (
            self.ipos >= len(self.filelist)
            if self._fetch_by_files
            else self.ipos >= self.load_range[1]
        )
        if end_of_list:
            if init:
                raise RuntimeError(
                    "Nothing to load for worker %d" % 0
                    if self.worker_info is None
                    else self.worker_info.id
                )
            if self._infinity_mode and not self._in_memory:
                # infinity mode: re-start
                self.restart()
                return
            else:
                # finite mode: set prefetch to None, exit
                self.prefetch = None
                return
        if self._fetch_by_files:
            filelist = self.filelist[int(self.ipos) : int(self.ipos + self._fetch_step)]
            load_range = self.load_range
        else:
            filelist = self.filelist
            load_range = (
                self.ipos,
                min(self.ipos + self._fetch_step, self.load_range[1]),
            )
        # _logger.info('Start fetching next batch, len(filelist)=%d, load_range=%s'%(len(filelist), load_range))
        if self._async_load:
            self.prefetch = self.executor.submit(
                _load_next,
                self._data_config,
                filelist,
                load_range,
                self._sampler_options,
            )
        else:
            self.prefetch = _load_next(
                self._data_config, filelist, load_range, self._sampler_options
            )
        self.ipos += self._fetch_step

    def get_data(self, i):
        # inputs
        X = {k: self.table["_" + k][i].copy() for k in self._data_config.input_names}
        if "EFlowPhoton" in X:
            return create_jets_outputs_Delphes(X), False
        return create_jets_outputs_new(X), False

class EventDatasetCollection(torch.utils.data.Dataset):
    def __init__(self, dir_list, args, aug_soft=False, aug_collinear=False):
        self.event_collections_dict = OrderedDict()
        for dir in dir_list:
            self.event_collections_dict[dir] = EventDataset.from_directory(dir, mmap=True, aug_soft=args.augment_soft_particles or aug_soft, seed=0, aug_collinear=aug_collinear)
        self.n_events = sum([x.n_events for x in self.event_collections_dict.values()])
        self.event_thresholds = [x.n_events for x in self.event_collections_dict.values()]
        self.event_thresholds = np.cumsum([0] + self.event_thresholds)
        self.dir_list = dir_list
    def __len__(self):
        return self.n_events
    def get_idx(self, i):
        assert i < self.n_events, "Index out of bounds: %d >= %d" % (i, self.n_events)
        for j in range(len(self.event_thresholds)-1):
            threshold = self.event_thresholds[j]
            if i >= threshold and i < self.event_thresholds[j+1]:
                #print("-------------", i, threshold, self.event_thresholds, j, self.dir_list[j])
                return self.event_collections_dict[self.dir_list[j]][i - threshold]
    def getitem(self, i):
        return self.get_idx(i)
    def __iter__(self):
        for i in range(self.n_events):
            yield self.get_idx(i)
    def __getitem__(self, i):
        assert i < self.n_events, "Index out of bounds: %d >= %d" % (i, self.n_events)
        return self.get_idx(i)
    # A collection of EventDatasets.
    # You should use a sampler together with this, as by default it just concatenates the EventDatasets together!

def get_batch_bounds(batch_idx):
    # batch_idx: tensor of format [0,0,0,0,1,1,1...]
    # returns tensor of format [0, 4, ...]
    print("Batch idx", batch_idx.shape, batch_idx[(batch_idx>3130) & (batch_idx < 3140)])
    batches = sorted(batch_idx.unique().tolist())
    skipped = []
    for i in range(batch_idx.max().int().item()):
        if i not in batches:
            skipped.append(i)
    # reverse sort skipped
    skipped = sorted(skipped, reverse=True)
    result = torch.zeros(batch_idx.max().int().item() + 2 + len(skipped))
    #for i, b in enumerate(batches):
    #    assert i == b
    #    result[i] = torch.where(batch_idx==b)[0].min()
    #    result[i+1] = torch.where(batch_idx==b)[0].max()
    b_list = batch_idx.int().tolist()
    prev = -1
    for i, b in enumerate(b_list):
        if b != prev:
            result[b] = i
            prev = b
    result[-1] = len(b_list)
    print("skipped", skipped)
    for s in skipped:
        if s == 0:
            result[s] = 0
        else:
            result[s] = result[s+1]
    print("result", result.shape, result[3130:3140].tolist())
    return result


def filter_pfcands(pfcands):
    # filter the GenParticles so that dark matter particles are not present
    # dark matter particles are defined as those with abs(pdgId) > 10000 or pdgId between 50-60
    # TODO: filter out high eta - temporarily this is done here, but it should be done in the ntuplizer in order to avoid big files
    mask = (torch.abs(pfcands.pid) < 10000) & ((torch.abs(pfcands.pid) < 50) | (torch.abs(pfcands.pid) > 60)) & (torch.abs(pfcands.eta) < 2.4) & (pfcands.pt > 0.5)#& (pfcands.pt > 0.5)
    pfcands.mask(mask)
    return pfcands

class EventDataset(torch.utils.data.Dataset):
    @staticmethod
    def from_directory(dir, mmap=True, model_clusters_file=None, model_output_file=None, include_model_jets_unfiltered=False, fastjet_R=None, parton_level=False, gen_level=False, aug_soft=False, seed=0, aug_collinear=False):
        result = {}
        for file in os.listdir(dir):
            if file == "metadata.pkl":
                metadata = pickle.load(open(os.path.join(dir, file), "rb"))
            else:
                result[file.split(".")[0]] = np.load(
                    os.path.join(dir, file), mmap_mode="r" if mmap else None
                )
        dataset = EventDataset(result, metadata, model_clusters_file=model_clusters_file,
                               model_output_file=model_output_file,
                               include_model_jets_unfiltered=include_model_jets_unfiltered,
                               fastjet_R=fastjet_R, parton_level=parton_level, gen_level=gen_level, aug_soft=aug_soft,
                               seed=seed, aug_collinear=aug_collinear)
        return dataset
    def get_pfcands_key(self):
        pfcands_key = "pfcands"
        print("get_pfcands_key")
        if self.gen_level:
            return "final_gen_particles"
        if self.parton_level:
            return "final_parton_level_particles"
        if self.model_output is None:
            if self.gen_level:
                return "final_gen_particles"
            if self.parton_level:
                return "final_parton_level_particles"
            return pfcands_key # ignore
        for i in [0, 1, 2]: # try the first three if it fits
            start = {key: self.metadata[key + "_batch_idx"][i] for key in self.attrs}
            end = {key: self.metadata[key + "_batch_idx"][i + 1] for key in self.attrs}
            result = {key: self.events[key][start[key]:end[key]] for key in self.attrs}
            result = {key: EventCollection.deserialize(result[key], batch_number=None, cls=Event.evt_collections[key])
                      for key in self.attrs}
            if "final_parton_level_particles" in result:
                result["final_parton_level_particles"] = filter_pfcands(result["final_parton_level_particles"])
            if "final_gen_particles" in result:
                result["final_gen_particles"] = filter_pfcands(result["final_gen_particles"])
            event_filter_s, event_filter_e = self.model_output["event_idx_bounds"][i].int().item(), \
            self.model_output["event_idx_bounds"][i + 1].int().item()
            diff = event_filter_e - event_filter_s
            if diff != len(result["pfcands"]):
                if diff == len(result["final_parton_level_particles"]):
                    pfcands_key = "final_parton_level_particles"
                    break
                if diff == len(result["final_gen_particles"]):
                    pfcands_key = "final_gen_particles"
                    break
        print("Found pfcands_key=%s" % pfcands_key)
        return pfcands_key

    def __init__(self, events, metadata, model_clusters_file=None, model_output_file=None, include_model_jets_unfiltered=False, fastjet_R=None, parton_level=False, gen_level=False, aug_soft=False, seed=0, aug_collinear=False):
        # events: serialized events dict
        # metadata: dict with metadata
        self.events = events
        self.n_events = metadata["n_events"]
        self.attrs = metadata["attrs"]
        self.metadata = metadata
        self.include_model_jets_unfiltered = include_model_jets_unfiltered
        self.model_i = 0
        self.parton_level = parton_level
        self.gen_level = gen_level
        self.augment_soft_particles = aug_soft
        self.aug_collinear = aug_collinear
        self.seed = seed
        #self.pfcands_key = "pfcands"
        # set to final_parton_level_particles or final_gen_particles in case needed
        #for key in self.attrs:
        #    self.evt_idx_to_batch_idx[key] = {}
        if model_output_file is not None:
            if type(model_output_file) == str:
                self.model_output = CPU_Unpickler(open(model_output_file, "rb")).load()
            else:
                self.model_output = model_output_file
            self.model_output["event_idx_bounds"] = get_batch_bounds(self.model_output["event_idx"])
            self.n_events = self.model_output["event_idx"].max().int().item()  # sometimes the last batch gets cut off, which causes problems
            if model_clusters_file is not None:
                self.model_clusters = to_tensor(pickle.load((open(model_clusters_file, "rb"))))
            else:
                self.model_clusters = self.model_output["model_cluster"]
            # model_output["batch_idx"] contains the batch index for each event. model_clusters is an array of the model labels for each event.
        else:
            self.model_output = None
            self.model_clusters = None
        if fastjet_R is not None:
            self.fastjet_jetdef = {r: fastjet.JetDefinition(fastjet.antikt_algorithm, r) for r in fastjet_R}
            ## fastjet_R is an array of radiuses for which to compute that

        self.pfcands_key = self.get_pfcands_key()
    def __len__(self):
        return self.n_events
   # def __next__(self):
    def add_model_output(self, model_output):
        if model_output is not None:
            if type(model_output) == str:
                self.model_output = CPU_Unpickler(open(model_output, "rb")).load()
            else:
                self.model_output = model_output
            self.model_output["event_idx_bounds"] = get_batch_bounds(self.model_output["event_idx"])
            self.n_events = self.model_output["event_idx"].max().int().item()  # sometimes the last batch gets cut off, which causes problems
            self.model_clusters = self.model_output["model_cluster"]
            # model_output["batch_idx"] contains the batch index for each event. model_clusters is an array of the model labels for each event.
        else:
            self.model_output = None
            self.model_clusters = None

    @staticmethod
    def pfcands_add_soft_particles(pfcands, n_soft, random_generator, add_original_particle_mapping=False):
        # augment the dataset with soft particles
        eta_bounds = [-2.4, 2.4]
        phi_bounds = [-3.14, 3.14]
        #pt_bounds = [0.02, 0.5]
        # choose random eta and phi
        # use the random generator for eta, phi
        eta = random_generator.uniform(eta_bounds[0], eta_bounds[1], n_soft).astype(np.double)
        phi = random_generator.uniform(phi_bounds[0], phi_bounds[1], n_soft).astype(np.double)
        #pt = random_generator.uniform(pt_bounds[0], pt_bounds[1], n_soft).astype(np.double)
        pt = np.ones(n_soft).astype(np.double) * 1e-2
        charge = np.zeros(n_soft).astype(np.double)
        pid = np.zeros(n_soft).astype(np.double)
        mass = np.zeros(n_soft).astype(np.double)
        if hasattr(pfcands, "status"):
            status = np.zeros(n_soft)
            soft_pfcands = EventPFCands(pt, eta, phi, mass, charge, pid, pf_cand_jet_idx=-1 * torch.ones(n_soft), status=status)
        else:
            soft_pfcands = EventPFCands(pt, eta, phi, mass, charge, pid, pf_cand_jet_idx=-1*torch.ones(n_soft))
        soft_pfcands.original_particle_mapping = torch.tensor([-1] * len(soft_pfcands))
        pfcandsc = copy.deepcopy(pfcands)
        pfcandsc.original_particle_mapping = torch.arange(len(pfcands))
        pfcandsc = concat_event_collection([pfcandsc, soft_pfcands], nobatch=1)
        if not add_original_particle_mapping:
            pfcandsc.original_particle_mapping = torch.arange(len(pfcandsc)) # for now, ignore the soft particles
        return pfcandsc

    @staticmethod
    def pfcands_split_particles(pfcands, random_generator):
        # Augment the dataset by spliting the harder particles
        # 5 highest pt particles
        k = min(5, len(pfcands))
        highest_pt_idx = torch.topk(pfcands.pt, k)[1]
        weights = pfcands.pt[highest_pt_idx]
        # Pick a random particle to split according to weights
        n_to_split = random_generator.randint(0, k)
        #idx = random_generator.choice(highest_pt_idx, p=weights / weights.sum())
        indices = highest_pt_idx[:n_to_split]
        pfcandsc = copy.deepcopy(pfcands)
        pfcandsc.original_particle_mapping = torch.arange(len(pfcands))
        for idx in indices:
            split_into = random_generator.randint(2, 5)
            # split the particle into
            eta = pfcands.eta[idx]
            phi = pfcands.phi[idx]
            pt = pfcands.pt[idx] / split_into
            charge = pfcands.charge[idx]
            mass = 0
            pid = pfcands.pid[idx]
            colinear_pfcands = EventPFCands(pt=[pt], eta=[eta], phi=[phi], mass=[mass], charge=[charge], pid=[pid], pf_cand_jet_idx=[pfcands.pf_cand_jet_idx[idx]], original_particle_mapping=[idx])
            #pfcandsc.original_particle_mapping[idx] = idx
            pfcandsc.pt[idx] = pt
            for _ in range(split_into-1):
                pfcandsc = concat_event_collection([pfcandsc, colinear_pfcands], nobatch=1)
        return pfcandsc

    def get_idx(self, i):
        start = {key: self.metadata[key + "_batch_idx"][i] for key in self.attrs}
        end = {key: self.metadata[key + "_batch_idx"][i + 1] for key in self.attrs}
        result = {key: self.events[key][start[key]:end[key]] for key in self.attrs}
        result = {key: EventCollection.deserialize(result[key], batch_number=None, cls=Event.evt_collections[key]) for
                  key in self.attrs}
        if "final_parton_level_particles" in result:
            #print("i=", i)
            #print("BEFORE:", len(result["final_parton_level_particles"]))
            result["final_parton_level_particles"] = filter_pfcands(result["final_parton_level_particles"])
            #print("AFTER:", len(result["final_parton_level_particles"]))
            #print("------")
        if "final_gen_particles" in result:
            result["final_gen_particles"] = filter_pfcands(result["final_gen_particles"])
        ## augment pfcands here
        if self.augment_soft_particles:
            random_generator = np.random.RandomState(seed=i + self.seed)
            #n_soft = int(random_generator.uniform(10, 1000))
            n_soft = 500
            #n_soft = 1000
            result["pfcands"] = EventDataset.pfcands_add_soft_particles(result["pfcands"], n_soft, random_generator)
            if "final_parton_level_particles" in result:
                result["final_parton_level_particles"] = EventDataset.pfcands_add_soft_particles(result["final_parton_level_particles"], n_soft, random_generator) # Also augment parton-level event for testing
            if "final_gen_particles" in result:
                result["final_gen_particles"] = EventDataset.pfcands_add_soft_particles(result["final_gen_particles"], n_soft, random_generator)
        else:
            result["pfcands"].original_particle_mapping = torch.arange(len(result["pfcands"].pt))
        if self.aug_collinear:
            random_generator = np.random.RandomState(seed=i + self.seed)
            result["pfcands"] = EventDataset.pfcands_split_particles(result["pfcands"], random_generator)
            if "final_parton_level_particles" in result:
                result["final_parton_level_particles"] = EventDataset.pfcands_split_particles(
                    result["final_parton_level_particles"], random_generator
                )
                # Also augment parton-level event for testing
            if "final_gen_particles" in result:
                result["final_gen_particles"] = EventDataset.pfcands_split_particles(result["final_gen_particles"], random_generator)
        if self.model_output is not None:
            #if "final_parton_level_particles" in result and len(result["final_parton_level_particles"]) == 0:
            #    print("!!")
            #    return None
            result["model_jets"], bc_scores_pfcands, bc_labels_pfcands = self.get_model_jets(i, pfcands=result[self.pfcands_key], include_target=1, dq=result["matrix_element_gen_particles"])
            result[self.pfcands_key].bc_scores_pfcands = bc_scores_pfcands
            result[self.pfcands_key].bc_labels_pfcands = bc_labels_pfcands
            if self.include_model_jets_unfiltered:
                result["model_jets_unfiltered"], _, _ = self.get_model_jets(i, pfcands=result[self.pfcands_key], filter=False)
        if hasattr(self, "fastjet_jetdef") and self.fastjet_jetdef is not None:
            if self.gen_level:
                result["fastjet_jets"] = {key: EventDataset.get_fastjet_jets(result, self.fastjet_jetdef[key], key="final_gen_particles") for key in self.fastjet_jetdef}
            elif self.parton_level:
                result["fastjet_jets"] = {key: EventDataset.get_fastjet_jets(result, self.fastjet_jetdef[key], key="final_parton_level_particles") for key in self.fastjet_jetdef}
            else:
                result["fastjet_jets"] = {key: EventDataset.get_fastjet_jets(result, self.fastjet_jetdef[key], key="pfcands") for key
                                          in self.fastjet_jetdef}
        if "genjets" in result:
            result["genjets"] = EventDataset.mask_jets(result["genjets"])
        evt = Event(**result)
        return evt

    @staticmethod
    def get_target_obj_score(clusters_eta, clusters_phi, clusters_pt, event_idx_clusters, dq_eta, dq_phi, dq_event_idx):
        # return the target scores for each cluster (reteurns list of 1's and 0's)
        # dq_coords: list of [eta, phi] for each dark quark
        # dq_event_idx: list of event_idx for each dark quarks
        target = []
        for event in event_idx_clusters.unique():
            filt = event_idx_clusters == event
            clusters = torch.stack([clusters_eta[filt], clusters_phi[filt], clusters_pt[filt]], dim=1)
            dq_coords_event = torch.stack([dq_eta[dq_event_idx == event], dq_phi[dq_event_idx == event]], dim=1)
            dist_matrix = torch.cdist(
                dq_coords_event,
                clusters[:, :2].to(dq_coords_event.device),
                p=2
            ).T
            if len(dist_matrix) == 0:
                target.append(torch.zeros(len(clusters)).int().to(dist_matrix.device))
                continue
            closest_quark_dist, closest_quark_idx = dist_matrix.min(dim=1)
            closest_quark_idx[closest_quark_dist > 0.8] = -1
            target.append((closest_quark_idx != -1).float())
        if len(target):
            return torch.cat(target).flatten()
        return torch.tensor([])

    @staticmethod
    def mask_jets(jets, cutoff=100):
        mask = jets.pt >= cutoff
        return EventJets(jets.pt[mask], jets.eta[mask], jets.phi[mask], jets.mass[mask])

    @staticmethod
    def get_model_jets_static(i, pfcands, model_output, model_clusters):
        event_filter_s, event_filter_e = model_output["event_idx_bounds"][i].int().item(), model_output["event_idx_bounds"][i + 1].int().item()
        pfcands_pt = pfcands.pt
        pfcands_pxyz = pfcands.pxyz
        pfcands_E = pfcands.E
        #assert len(pfcands_pt) == event_filter_e - event_filter_s, "Error!, len(pfcands_pt)==%d, event_filter_e-event_filter_s=%d" % (len(pfcands_pt), event_filter_e - event_filter_s)
        if not len(pfcands_pt) == event_filter_e - event_filter_s:
            return None
        # jets_pt = scatter_sum(to_tensor(pfcands_pt), self.model_clusters[event_filter] + 1, dim=0)[1:]
        jets_pxyz = scatter_sum(to_tensor(pfcands_pxyz), model_clusters[event_filter_s:event_filter_e] + 1, dim=0)[1:]
        jets_pt = torch.norm(jets_pxyz[:, :2], p=2, dim=-1)
        jets_eta, jets_phi = calc_eta_phi(jets_pxyz, False)
        # jets_mass = torch.zeros_like(jets_eta)
        jets_E = scatter_sum(to_tensor(pfcands_E), model_clusters[event_filter_s:event_filter_e] + 1, dim=0)[1:]
        jets_mass = torch.sqrt(jets_E ** 2 - jets_pxyz.norm(dim=-1) ** 2)
        cluster_labels = model_clusters[event_filter_s:event_filter_e]
        bc_scores = model_output["pred"][event_filter_s:event_filter_e, -1]
        cutoff = 100
        mask = jets_pt >= cutoff
        return EventJets(jets_pt[mask], jets_eta[mask], jets_phi[mask], jets_mass[mask])

    @staticmethod
    def get_jets_fastjets_raw(pfcands, jetdef):
        pt = []
        eta = []
        phis = []
        mass = []
        array = get_pseudojets_fastjet(pfcands)
        cluster = fastjet.ClusterSequence(array, jetdef)
        inc_jets = cluster.inclusive_jets()
        for elem in inc_jets:
            if elem.pt() < 100:
                continue
            # print("pt:", elem.pt(), "eta:", elem.rap(), "phi:", elem.phi())ž
            pt.append(elem.pt())
            eta.append(elem.rap())
            phi = elem.phi()
            if phi > np.pi:
                phi -= 2 * np.pi
            phis.append(phi)
            mass.append(elem.m())
        return pt, eta, phis, mass

    @staticmethod
    def get_fastjet_jets(event, jetdef, key="pfcands"):
        pt, eta, phi, m = EventDataset.get_jets_fastjets_raw(getattr(event, key), jetdef)
        return EventJets(torch.tensor(pt), torch.tensor(eta), torch.tensor(phi), torch.tensor(m))

    def get_model_jets(self, i, pfcands, filter=True, dq=None, include_target=False):
        event_filter_s, event_filter_e = self.model_output["event_idx_bounds"][i].int().item(), self.model_output["event_idx_bounds"][i+1].int().item()
        pfcands_pt = pfcands.pt
        pfcands_pxyz = pfcands.pxyz
        pfcands_E = pfcands.E
        obj_score = None
        #print("Len pfcands_pt", len(pfcands_pt), "event_filter_e", event_filter_e, "event_filter_s", event_filter_s)
        if len(pfcands_pt) == 0:
            return EventJets(torch.tensor([]), torch.tensor([]), torch.tensor([]) ,torch.tensor([])), None, None
        assert len(pfcands_pt) == event_filter_e - event_filter_s, "Error! filter={} len(pfcands_pt)={} event_filter_e={} event_filter_s={}".format(filter, len(pfcands_pt), event_filter_e, event_filter_s)
        #jets_pt = scatter_sum(to_tensor(pfcands_pt), self.model_clusters[event_filter] + 1, dim=0)[1:]
        jets_pxyz = scatter_sum(to_tensor(pfcands_pxyz), self.model_clusters[event_filter_s:event_filter_e] + 1, dim=0)[1:]
        jets_pt = torch.norm(jets_pxyz[:, :2], p=2, dim=-1)
        jets_eta, jets_phi = calc_eta_phi(jets_pxyz, False)
        #jets_mass = torch.zeros_like(jets_eta)
        jets_E = scatter_sum(to_tensor(pfcands_E), self.model_clusters[event_filter_s:event_filter_e] + 1, dim=0)[1:]
        jets_mass = torch.sqrt(jets_E**2 - jets_pxyz.norm(dim=-1)**2)
        cluster_labels = self.model_clusters[event_filter_s:event_filter_e]
        bc_scores = self.model_output["pred"][event_filter_s:event_filter_e, -1]
        if "obj_score_pred" in self.model_output and not torch.is_tensor(self.model_output["obj_score_pred"]):
            self.model_output["obj_score_pred"] = torch.cat(self.model_output["obj_score_pred"])
            print("Concatenated obj_score_pred")
        target_obj_score = None
        if filter:
            cutoff = 100
            mask = jets_pt >= cutoff
            if "obj_score_pred" in self.model_output:
                obj_score = self.model_output["obj_score_pred"][(self.model_output["event_clusters_idx"] == i)]
                #print("Jets pt", jets_pt, "obj score", obj_score)
                assert len(obj_score) == len(jets_pt), "Error! len(obj_score)=%d, len(jets_pt)=%d" % (
                len(obj_score), len(jets_pt))
                if include_target:
                    target_obj_score = EventDataset.get_target_obj_score(jets_eta, jets_phi, jets_pt, torch.zeros(jets_pt.size(0)), dq.eta, dq.phi, torch.zeros(dq.eta.size(0)))
        else:
            mask = torch.ones_like(jets_pt, dtype=torch.bool)
        if obj_score is not None:
            obj_score = obj_score[mask]
            assert len(jets_pt[mask]) == len(obj_score), "Error! len(jets_pt[mask])=%d, len(obj_score)=%d" % (len(jets_pt[mask]), len(obj_score))
        if target_obj_score is not None:
            target_obj_score = target_obj_score[mask]
            assert len(jets_pt[mask]) == len(target_obj_score), "Error! len(jets_pt[mask])=%d, len(obj_score)=%d" % (len(jets_pt[mask]), len(obj_score))
        return EventJets(jets_pt[mask], jets_eta[mask], jets_phi[mask], jets_mass[mask], obj_score=obj_score, target_obj_score=target_obj_score), bc_scores, cluster_labels
    def get_iter(self):
        self.i = 0
        while self.i < self.n_events:
            yield self.get_idx(self.i)
            self.i += 1
    def __iter__(self):
        return self.get_iter()
    def __getitem__(self, i):
        assert i < self.n_events, "Index out of bounds: %d >= %d" % (i, self.n_events)
        return self.get_idx(i)


class SimpleIterDataset(torch.utils.data.IterableDataset):
    r"""Base IterableDataset.
    Handles dataloading.
    Arguments:
        file_dict (dict): dictionary of lists of files to be loaded.
        data_config_file (str): YAML file containing data format information.
        for_training (bool): flag indicating whether the dataset is used for training or testing.
            When set to ``True``, will enable shuffling and sampling-based reweighting.
            When set to ``False``, will disable shuffling and reweighting, but will load the observer variables.
        load_range_and_fraction (tuple of tuples, ``((start_pos, end_pos), load_frac)``): fractional range of events to load from each file.
            E.g., setting load_range_and_fraction=((0, 0.8), 0.5) will randomly load 50% out of the first 80% events from each file (so load 50%*80% = 40% of the file).
        fetch_by_files (bool): flag to control how events are retrieved each time we fetch data from disk.
            When set to ``True``, will read only a small number (set by ``fetch_step``) of files each time, but load all the events in these files.
            When set to ``False``, will read from all input files, but load only a small fraction (set by ``fetch_step``) of events each time.
            Default is ``False``, which results in a more uniform sample distribution but reduces the data loading speed.
        fetch_step (float or int): fraction of events (when ``fetch_by_files=False``) or number of files (when ``fetch_by_files=True``) to load each time we fetch data from disk.
            Event shuffling and reweighting (sampling) is performed each time after we fetch data.
            So set this to a large enough value to avoid getting an imbalanced minibatch (due to reweighting/sampling), especially when ``fetch_by_files`` set to ``True``.
            Will load all events (files) at once if set to non-positive value.
        file_fraction (float): fraction of files to load.
    """

    def __init__(
        self,
        file_dict,
        data_config_file,
        for_training=True,
        load_range_and_fraction=None,
        extra_selection=None,
        fetch_by_files=False,
        fetch_step=0.01,
        file_fraction=1,
        remake_weights=False,
        up_sample=True,
        weight_scale=1,
        max_resample=10,
        async_load=True,
        infinity_mode=False,
        in_memory=False,
        name="",
        laplace=False,
        edges=False,
        diffs=False,
        dataset_cap=None,
        n_noise=0,
        synthetic=False,
        synthetic_npart_min=2,
        synthetic_npart_max=5,
        jets=False,
    ):
        self._iters = {} if infinity_mode or in_memory else None
        _init_args = set(self.__dict__.keys())
        self._init_file_dict = file_dict
        self._init_load_range_and_fraction = load_range_and_fraction
        self._fetch_by_files = fetch_by_files
        self._fetch_step = fetch_step
        self._file_fraction = file_fraction
        self._async_load = async_load
        self._infinity_mode = infinity_mode
        self._in_memory = in_memory
        self._name = name
        self.laplace = laplace
        self.edges = edges
        self.diffs = diffs
        self.synthetic = synthetic
        self.synthetic_npart_min = synthetic_npart_min
        self.synthetic_npart_max = synthetic_npart_max
        self.dataset_cap = dataset_cap  # used to cap the dataset to some fixed number of events - used for debugging purposes
        self.n_noise = n_noise
        self.jets = jets
        # ==== sampling parameters ====
        self._sampler_options = {
            "up_sample": up_sample,
            "weight_scale": weight_scale,
            "max_resample": max_resample,
        }

        if for_training:
            self._sampler_options.update(training=True, shuffle=False, reweight=True)
        else:
            self._sampler_options.update(training=False, shuffle=False, reweight=False)

        # discover auto-generated reweight file
        if ".auto.yaml" in data_config_file:
            data_config_autogen_file = data_config_file
        else:
            data_config_md5 = _md5(data_config_file)
            data_config_autogen_file = data_config_file.replace(
                ".yaml", ".%s.auto.yaml" % data_config_md5
            )
            if os.path.exists(data_config_autogen_file):
                data_config_file = data_config_autogen_file
                _logger.info(
                    "Found file %s w/ auto-generated preprocessing information, will use that instead!"
                    % data_config_file
                )

        # load data config (w/ observers now -- so they will be included in the auto-generated yaml)
        self._data_config = DataConfig.load(data_config_file)

        if for_training:
            # produce variable standardization info if needed
            if self._data_config._missing_standardization_info:
                s = AutoStandardizer(file_dict, self._data_config)
                self._data_config = s.produce(data_config_autogen_file)

            # produce reweight info if needed
            # if self._sampler_options['reweight'] and self._data_config.weight_name and not self._data_config.use_precomputed_weights:
            #    if remake_weights or self._data_config.reweight_hists is None:
            #        w = WeightMaker(file_dict, self._data_config)
            #        self._data_config = w.produce(data_config_autogen_file)

            # reload data_config w/o observers for training
            if (
                os.path.exists(data_config_autogen_file)
                and data_config_file != data_config_autogen_file
            ):
                data_config_file = data_config_autogen_file
                _logger.info(
                    "Found file %s w/ auto-generated preprocessing information, will use that instead!"
                    % data_config_file
                )
            self._data_config = DataConfig.load(
                data_config_file, load_observers=False, extra_selection=extra_selection
            )
        else:
            self._data_config = DataConfig.load(
                data_config_file,
                load_reweight_info=False,
                extra_test_selection=extra_selection,
            )

        # Derive all variables added to self.__dict__
        self._init_args = set(self.__dict__.keys()) - _init_args

    @property
    def config(self):
        return self._data_config

    def __iter__(self):
        if self._iters is None:
            kwargs = {k: copy.deepcopy(self.__dict__[k]) for k in self._init_args}
            return _SimpleIter(**kwargs)
        else:
            worker_info = torch.utils.data.get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            try:
                return self._iters[worker_id]
            except KeyError:
                kwargs = {k: copy.deepcopy(self.__dict__[k]) for k in self._init_args}
                self._iters[worker_id] = _SimpleIter(**kwargs)
                return self._iters[worker_id]
