import numpy as np
from sklearn.cluster import DBSCAN


def clusterize(points, radius):
    # radius = ROAD.WIDTH * 0.4
    clusterize.dbscan = DBSCAN(eps=radius,
                               min_samples=1,
                               metric='euclidean',
                               algorithm='kd_tree',
                               leaf_size=10)
    return clusterize.dbscan.fit_predict(np.array([[p[0], p[1]] for p in points]))


def unite_noise(clusters, noise_len=1):
    new_labels = {}
    for label, count in zip(*np.unique(clusters, return_counts=True)):
        if count <= noise_len:
            new_labels[label] = -1
        else:
            new_labels[label] = label

    result = []
    for label in clusters:
        result.append(new_labels[label])
    return result
