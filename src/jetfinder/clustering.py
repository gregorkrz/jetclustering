import hdbscan
from tqdm import tqdm
import numpy as np
import torch

def lorentz_norm_comp(vec1, vec2):
    diff = vec1-vec2
    norm_squared = np.abs(diff[0]**2 - diff[1]**2 - diff[2] ** 2 - diff[3]**2)
    return np.sqrt(norm_squared)

def get_distance_matrix(v):
    # compute the cosine similarity between vectors in matrix, fast format
    # v is a numpy array
    # returns a numpy array
    if torch.is_tensor(v):
        v = v.double().numpy()
    dot_product = np.dot(v, v.T)
    magnitude = np.sqrt(np.sum(np.square(v), axis=1))
    magnitude = magnitude[:, np.newaxis]
    return dot_product / (magnitude * magnitude.T)

def get_distance_matrix_Lorentz(v):
    # Lorentz cosine similarity distance metric
    # Lorentz dot product:
    if torch.is_tensor(v):
        v = v.double().numpy()
    dot_product = np.outer(v[:, 0], v[:, 0])  - np.outer(v[:, 1], v[:, 1]) - np.outer(v[:, 2], v[:, 2]) - np.outer(v[:, 3], v[:, 3])
    #magnitude = np.sqrt(np.abs(np.sum(np.square(v), axis=1)))
    # lorentz magnitude
    magnitude = np.sqrt(np.abs(v[:, 0]**2 - v[:, 1]**2 - v[:, 2] ** 2 - v[:, 3]**2))
    magnitude = magnitude[:, np.newaxis]
    return dot_product #/ (magnitude * magnitude.T)


def get_clustering_labels(coords, batch_idx, min_cluster_size=10, min_samples=20, epsilon=0.1, bar=False,
                          lorentz_cos_sim=False, cos_sim=False, return_labels_event_idx=False):
    # return_labels_event_idx: If True, it will return the labels with unique numbers and event_idx tensor for each label
    labels = []
    it = np.unique(batch_idx)
    labels_event_idx = []
    max_cluster_idx = 0
    count = 0
    if bar:
        it = tqdm(it)
    for i in it:
        filt = batch_idx == i
        c = coords[filt]
        kwargs = {}
        if lorentz_cos_sim:
            kwargs["metric"] = "precomputed"
            c = get_distance_matrix_Lorentz(c)
            #print(c)
        elif cos_sim:
            kwargs["metric"] = "precomputed"
            c = get_distance_matrix(c)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                    cluster_selection_epsilon=epsilon, **kwargs)
        cluster_labels = clusterer.fit_predict(c)
        if return_labels_event_idx:
            num_clusters = np.max(cluster_labels) + 1
            labels_event_idx.append([count] * (num_clusters))
            count += 1
            cluster_labels += max_cluster_idx
            max_cluster_idx += num_clusters
        labels.append(cluster_labels)
    assert len(np.concatenate(labels)) == len(coords)
    if return_labels_event_idx:
        return np.concatenate(labels), np.concatenate(labels_event_idx)
    return np.concatenate(labels)
