import hdbscan
from tqdm import tqdm
import numpy as np
import torch
from sklearn.cluster import DBSCAN

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

def custom_metric(xyz, pt):
    """
    Computes the distance matrix where the distance function is defined as:
    Euclidean distance between two points in xyz space * min(pt1, pt2)

    Parameters:
    xyz (numpy.ndarray): An (N, 3) array of N points in 3D space.
    pt (numpy.ndarray): A (N,) array of scalars associated with each point.

    Returns:
    numpy.ndarray: An (N, N) distance matrix.
    """
    N = xyz.shape[0]
    print("Len", N)
    distance_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                euclidean_distance = np.linalg.norm(xyz[i] - xyz[j])
                scale_factor = min(pt[i], pt[j])
                distance_matrix[i, j] = euclidean_distance * scale_factor

    return distance_matrix

def get_clustering_labels(coords, batch_idx, min_cluster_size=10, min_samples=20, epsilon=0.1, bar=False,
                          lorentz_cos_sim=False, cos_sim=False, return_labels_event_idx=False, pt=None):
    # return_labels_event_idx: If True, it will return the labels with unique numbers and event_idx tensor for each label
    labels = []
    labels_no_reindex = []
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
        elif pt is not None:
            kwargs["metric"] = "precomputed"
            c = custom_metric(c, pt)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                    cluster_selection_epsilon=epsilon, **kwargs)
        try:
            cluster_labels = clusterer.fit_predict(c)
        except Exception as e:
            print("Error in clustering", e)
            print("Coords", c.shape)
            print("Batch idx", batch_idx.shape)
            print("Setting the labels to -1")
            cluster_labels = np.full(len(c), -1)
        labels_no_reindex.append(cluster_labels)
        if return_labels_event_idx:
            num_clusters = np.max(cluster_labels) + 1
            labels_event_idx.append([count] * (num_clusters))
            count += 1
            cluster_labels += max_cluster_idx
            max_cluster_idx += num_clusters
        labels.append(cluster_labels)
    assert len(np.concatenate(labels)) == len(coords)
    if return_labels_event_idx:
        return np.concatenate(labels_no_reindex), np.concatenate(labels), np.concatenate(labels_event_idx)
    return np.concatenate(labels)


def get_clustering_labels_dbscan(coords, pt, batch_idx, min_samples=10, epsilon=0.1, bar=False, return_labels_event_idx=False):
    # return_labels_event_idx: If True, it will return the labels with unique numbers and event_idx tensor for each label
    labels = []
    labels_no_reindex = []
    it = np.unique(batch_idx)
    labels_event_idx = []
    max_cluster_idx = 0
    count = 0
    if bar:
        it = tqdm(it)
    for i in it:
        filt = batch_idx == i
        c = coords[filt]
        clusterer = DBSCAN(min_samples=min_samples, eps=epsilon)
        cluster_labels = clusterer.fit_predict(c, sample_weight=pt[filt])
        labels_no_reindex.append(cluster_labels)
        if return_labels_event_idx:
            num_clusters = np.max(cluster_labels) + 1
            labels_event_idx.append([count] * (num_clusters))
            count += 1
            cluster_labels += max_cluster_idx
            max_cluster_idx += num_clusters
        labels.append(cluster_labels)
    assert len(np.concatenate(labels)) == len(coords)
    if return_labels_event_idx:
        return np.concatenate(labels_no_reindex), np.concatenate(labels), np.concatenate(labels_event_idx)
    return np.concatenate(labels)

