import numpy as np
from sklearn.cluster import DBSCAN
from coordinates import ROAD


def clusterize(points):
    max_dist = ROAD.WIDTH * 0.4
    clusterize.dbscan = DBSCAN(eps=max_dist,
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


def update_labels(labels, clusters):
    clusters = unite_noise(clusters)

    p = 0
    for i in range(len(labels)):
        if labels[i] == 1:
            labels[i] = clusters[p]
            p += 1
