from tqdm import tqdm
import numpy as np
from src.dataset.dataset import EventDataset, get_batch_bounds

def compute_f1_score(dataset, dataset_cap=-1):
    R = 0.8
    counts = {
        "n_matched_dark_quarks": 0,
        "n_jets": 0,
        "n_dark_quarks": 0
    } # Array of [n_relevant_retrieved, all_retrieved, all_relevant], or in our language, [n_matched_dark_quarks, n_jets, n_dark_quarks]
    n = 0
    for x in tqdm(range(len(dataset))):
        data = dataset[x]
        jets_object = data.model_jets
        n += 1
        if dataset_cap != -1 and n >= dataset_cap:
            break
        jets = [jets_object.eta, jets_object.phi]
        dq = [data.matrix_element_gen_particles.eta, data.matrix_element_gen_particles.phi]
        # calculate deltaR between each jet and each quark
        distance_matrix = np.zeros((len(jets_object), len(data.matrix_element_gen_particles)))
        for i in range(len(jets_object)):
            for j in range(len(data.matrix_element_gen_particles)):
                deta = jets[0][i] - dq[0][j]
                dphi = jets[1][i] - dq[1][j]
                distance_matrix[i, j] = np.sqrt(deta**2 + dphi**2)
        # row-wise argmin
        distance_matrix = distance_matrix.T
        #min_distance = np.min(distance_matrix, axis=1)
        n_jets = len(jets_object)
        counts["n_jets"] += n_jets
        counts["n_dark_quarks"] += len(data.matrix_element_gen_particles)
        if len(jets_object):
            quark_to_jet = np.min(distance_matrix, axis=1)
            quark_to_jet[quark_to_jet > R] = -1
            counts["n_matched_dark_quarks"] += np.sum(quark_to_jet != -1)
    if counts["n_jets"] != 0:
        precision = counts["n_matched_dark_quarks"] / counts["n_jets"]
    else:
        precision = 0
    if counts["n_dark_quarks"] != 0:
        recall = counts["n_matched_dark_quarks"] / counts["n_dark_quarks"]
    else:
        recall = 0
    if precision == 0 or recall == 0:
        return 0
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score

def compute_f1_score_from_result(result, dataset):
    R = 0.8
    counts = {
        "n_matched_dark_quarks": 0,
        "n_jets": 0,
        "n_dark_quarks": 0
    } # Array of [n_relevant_retrieved, all_retrieved, all_relevant], or in our language, [n_matched_dark_quarks, n_jets, n_dark_quarks]
    result["event_idx_bounds"] = get_batch_bounds(result["event_idx"])
    l = result["event_idx"].max().int().item()
    for x in tqdm(range(len(dataset))):
        if x >= l:
            break
        data = dataset[x]
        jets_object = EventDataset.get_model_jets_static(x, data.pfcands, result, result["model_cluster"])
        if jets_object is None:
            continue
        jets = [jets_object.eta, jets_object.phi]
        dq = [data.matrix_element_gen_particles.eta, data.matrix_element_gen_particles.phi]
        # calculate deltaR between each jet and each quark
        distance_matrix = np.zeros((len(jets_object), len(data.matrix_element_gen_particles)))
        for i in range(len(jets_object)):
            for j in range(len(data.matrix_element_gen_particles)):
                deta = jets[0][i] - dq[0][j]
                dphi = jets[1][i] - dq[1][j]
                distance_matrix[i, j] = np.sqrt(deta**2 + dphi**2)
        # row-wise argmin
        distance_matrix = distance_matrix.T
        #min_distance = np.min(distance_matrix, axis=1)
        n_jets = len(jets_object)
        counts["n_jets"] += n_jets
        counts["n_dark_quarks"] += len(data.matrix_element_gen_particles)
        if len(jets_object):
            quark_to_jet = np.min(distance_matrix, axis=1)
            quark_to_jet[quark_to_jet > R] = -1
            counts["n_matched_dark_quarks"] += np.sum(quark_to_jet != -1)
    if counts["n_jets"] == 0 or counts["n_dark_quarks"] == 0:
        return 0
    precision = counts["n_matched_dark_quarks"] / counts["n_jets"]
    recall = counts["n_matched_dark_quarks"] / counts["n_dark_quarks"]
    f1_score = 2 * precision * recall / (precision + recall)
    return f1_score
