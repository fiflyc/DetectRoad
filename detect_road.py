import cv2
import numpy as np
from sklearn.neighbors import KDTree
from detect_tapes import detect_tapes
from coordinates import ROAD, set_horizon, image_to_plane, plane_to_image
from clusterization import clusterize, unite_noise
from lines import broken_line, extend_line, unite_lines, line_score


def detect_line(image):
    """
    Finds dash line on the image.
    :param image: input image.
    :return: dash line as numpy array.
    """

    image = np.copy(image)
    height, width, _ = image.shape
    set_horizon(height * 0.25)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find tapes:
    tapes, noise = detect_tapes(image)
    if len(tapes) == 0:
        return image
    noise_plane = image_to_plane(noise, (int(width / 2), int(height / 2)))
    tapes_plane = image_to_plane(tapes, (int(width / 2), int(height / 2)))

    # Clusterize tapes:
    tapes_clusters = clusterize(tapes_plane, ROAD.WIDTH * 0.5)
    tapes_clusters = unite_noise(tapes_clusters)

    np.append(noise, [tapes[i] for i in range(len(tapes)) if tapes_clusters == -1])
    np.append(noise_plane, [tapes_plane[i] for i in range(len(tapes_plane)) if tapes_clusters == -1])

    tapes = np.array([tapes[i] for i in range(len(tapes)) if tapes_clusters != -1])
    tapes_plane = np.array([tapes_plane[i] for i in range(len(tapes_plane)) if tapes_clusters != -1])

    # Remove groups of noise:
    noise_clusters = clusterize(noise_plane, ROAD.GAP)
    unite_noise(noise_clusters)

    noise_updated = [noise_plane[i] for i in range(len(noise_plane)) if noise_clusters[i] == -1]
    for c in range(max(noise_clusters)):
        center = np.average([np.array(noise_plane[i]) for i in range(len(noise_plane)) if noise_clusters[i] == c], axis=0)
        noise_updated.append(center)
    noise_updated = np.array([p for p in noise_updated if len(np.shape(p)) > 0])  # Remove NaNs

    noise_plane = noise_updated
    noise = plane_to_image(noise_plane, (int(width / 2), int(height / 2)))

    # Build broken line for each tapes cluster:
    lines_plane = []
    classes = np.unique(tapes_clusters)
    classes = np.delete(classes, np.argwhere(classes == -1))
    for c in classes:
        group = [i for i in range(len(tapes_plane)) if tapes_clusters[i] == c]
        line = broken_line(np.array([tapes_plane[i] for i in group]))
        lines_plane.append(np.array([tapes_plane[group[i]] for i in line]))

    if len(lines_plane) == 0:
        return None

    # Extend broken lines with noise:
    kd_tree = KDTree(noise_plane, 15)
    banned = []
    for i in range(len(lines_plane)):
        lines_plane[i] = extend_line(lines_plane[i], kd_tree, noise_plane, banned)

    # Unite and filter lines:
    lines_plane = unite_lines(lines_plane)
    best = plane_to_image(lines_plane[0], (int(width / 2), int(height / 2)))
    score = line_score(lines_plane[0], width, height)

    for line in lines_plane:
        line = plane_to_image(line, (int(width / 2), int(height / 2)))
        new_score = line_score(line, width, height)
        if score < new_score:
            best, score = line, new_score

    return best


def print_line(image, line):
    if line is None:
        return image

    for p, q in zip(line[:-1], line[1:]):
        p = (p[0], p[1])
        q = (q[0], q[1])
        cv2.line(image, p, q, (210, 195, 0), thickness=2)
        cv2.circle(image, p, 1, color=(195, 0, 50), thickness=4)
        cv2.circle(image, q, 1, color=(195, 0, 50), thickness=4)

    return image