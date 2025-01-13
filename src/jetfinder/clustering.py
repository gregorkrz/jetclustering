import hdbscan
from tqdm import tqdm
import numpy as np

def get_clustering_labels(coords, batch_idx, min_cluster_size=10, min_samples=20, epsilon=0.1, bar=False):
    labels = []
    it = np.unique(batch_idx)
    if bar:
        it = tqdm(it)
    for i in it:
        filt = batch_idx == i
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                                    cluster_selection_epsilon=epsilon)
        cluster_labels = clusterer.fit_predict(coords[filt])
        labels.append(cluster_labels)
    return np.concatenate(labels)
