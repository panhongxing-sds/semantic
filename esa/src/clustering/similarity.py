import numpy as np
from sklearn.cluster import DBSCAN, HDBSCAN


def dbscan_clustering(similarity_matrix: np.ndarray, clustering_threshold: float):
    """
    Cluster based on the similarity matrix with DBSCAN.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Pairwise similarity matrix.
    clustering_threshold : float
        Radius parameter for DBSCAN.
    """
    clustering = DBSCAN(
        eps=1 - clustering_threshold, 
        min_samples=1,
        metric="precomputed"
    )
    clustering.fit(1 - similarity_matrix)
    return clustering.labels_


def hdbscan_clustering(similarity_matrix: np.ndarray):
    """
    Cluster based on the similarity matrix with HDBSCAN.

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Pairwise similarity matrix.
    """
    clustering = HDBSCAN(
        metric="precomputed",
        allow_single_cluster=True,
        # min_cluster_size=1,
    )
    clustering.fit(1 - similarity_matrix)
    return clustering.labels_
