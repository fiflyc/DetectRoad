import os
import sys
import re
import cv2
import numpy as np
from sklearn.neighbors import KDTree
from detect_tapes import detect_tapes
from coordinates import ROAD, set_horizon, image_to_plane, plane_to_image
from clusterization import clusterize, unite_noise
from lines import broken_line, extend_line, unite_lines, line_score


def test():
    for filename in os.listdir('./input'):
        name, ext = os.path.splitext(filename)
        source = cv2.VideoCapture('./input/' + filename)
        result = cv2.VideoWriter('./output/' + name + "_detected" + ext,
                                 cv2.VideoWriter_fourcc(*'mp4v'),
                                 source.get(cv2.CAP_PROP_FPS),
                                 (int(source.get(3)), int(source.get(4))))

        height = source.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = source.get(cv2.CAP_PROP_FRAME_WIDTH)
        set_horizon(height * 0.25)

        frame = 0
        while source.isOpened():
            ret, image = source.read()
            if not ret:
                break
            frame += 1
            print("Frame", frame)

            image = modify_image(image, width, height)

            result.write(image)

        result.release()
        source.release()
        cv2.destroyAllWindows()


def main():
    with open(outfile, 'w') as out:
        pass


def modify_image(image, width, height):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find tapes:
    tapes, labels = detect_tapes(image)
    if len(tapes) == 0:
        return image
    noise = [tapes[i] for i in range(len(tapes)) if labels[i] == -1]
    tapes = [tapes[i] for i in range(len(tapes)) if labels[i] == 1]
    noise_plane = image_to_plane(noise, (int(width / 2), int(height / 2)))
    tapes_plane = image_to_plane(tapes, (int(width / 2), int(height / 2)))

    # Clusterize tapes:
    tapes_clusters = clusterize(tapes_plane, ROAD.WIDTH * 0.5)
    tapes_clusters = unite_noise(tapes_clusters)
    noise += [tapes[i] for i in range(len(tapes)) if tapes_clusters == -1]
    noise_plane += [tapes_plane[i] for i in range(len(tapes_plane)) if tapes_clusters == -1]
    tapes = [tapes[i] for i in range(len(tapes)) if tapes_clusters != -1]
    tapes_plane = [tapes_plane[i] for i in range(len(tapes_plane)) if tapes_clusters != -1]

    # Remove groups of noise:
    noise_clusters = clusterize(noise_plane, ROAD.GAP)
    unite_noise(noise_clusters)
    noise_updated = [noise_plane[i] for i in range(len(noise_plane)) if noise_clusters[i] == -1]
    for c in range(max(noise_clusters)):
        center = np.average([np.array(noise_plane[i]) for i in range(len(noise_plane)) if noise_clusters[i] == c], axis=0)
        noise_updated.append(center)
    noise_updated = [p for p in noise_updated if len(np.shape(p)) > 0]
    noise_plane = noise_updated
    noise = plane_to_image(noise_plane, (int(width / 2), int(height / 2)))

    # Build broken line for each tapes cluster:
    lines_plane = []
    for c in np.unique(tapes_clusters):
        if c != -1:
            group = [i for i in range(len(tapes_plane)) if tapes_clusters[i] == c]
            line = broken_line([tapes_plane[i] for i in group])
            lines_plane.append([list(tapes_plane[group[i]]) for i in line])

    # Extend broken lines with noise:
    kd_tree = KDTree(np.array(noise_plane), 15)
    banned = []
    for i in range(len(lines_plane)):
        lines_plane[i] = extend_line(lines_plane[i], kd_tree, noise_plane, banned)

    # Unite and filter lines:
    lines_plane = unite_lines(lines_plane)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for line in lines_plane:
        line_img = plane_to_image(line, (int(width / 2), int(height / 2)))
        for p, q in zip(line_img[:-1], line_img[1:]):
            cv2.line(image, p, q, (210, 195, 0), thickness=2)
            cv2.circle(image, p, 1, color=(195, 0, 50), thickness=4)
            cv2.circle(image, q, 1, color=(195, 0, 50), thickness=4)
        cv2.putText(image, str(line_score(line_img, width, height))[:4], line_img[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 190, 0))

    for point in noise:
        cv2.circle(image, point, 1, color=(50, 0, 195), thickness=4)
    for point in tapes:
        cv2.circle(image, point, 1, color=(195, 0, 50), thickness=4)

    return image


def print_angles(line):
    vectors = []
    for p, q in zip(line[:-1], line[1:]):
        vectors.append(np.array(q) - np.array(p))
    for u, v in zip(vectors[:-1], vectors[1:]):
        print(np.dot(u, v) / np.linalg.norm(u) / np.linalg.norm(v), end=" ")
    print(" ")


"""
mode:
    'main'  to find road in the input image. (default)
    'video' to get modified video showing script work.
            In this mode script gets all videos from './input' and saves the output in './output'. All directories must exist!
infile:
    Only in the 'main' mode!
    Video frame as image from robot camera.
outfile:
    Only in the 'main' mode!
    File with points of the road border.
"""


def parse_args():
    _mode = 'main'
    _infile = None
    _outfile = None

    for arg in sys.argv[1:]:
        var, value = re.split(r'=', arg)
        if var == '--mode':
            _mode = value
        elif var == '--in':
            _infile = value
        elif var == "--out":
            _outfile = value
        else:
            exit("No such option: " + var)

    return _mode, _infile, _outfile


if __name__ == '__main__':
    mode, infile, outfile = parse_args()
    if mode == 'video':
        test()
    elif mode == 'main':
        if infile is None or outfile is None:
            exit("Use --in= and --out= to choose input and output files")
        main()
    elif mode == 'pic':
        if infile is None or outfile is None:
            exit("Use --in= and --out= to choose input and output files")
        image = cv2.imread(infile)
        h, w, _ = image.shape
        set_horizon(h * 0.25)
        cv2.imwrite(outfile, modify_image(image, w, h))

