import os
import sys
import re
import cv2
import numpy as np
from sklearn.neighbors import KDTree
from detect_tapes import detect_tapes
from coordinates import set_horizon, image_to_plane, plane_to_image
from clusterization import clusterize, update_labels
from lines import broken_line, extend_line


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
    tapes_plane = image_to_plane(tapes, (int(width / 2), int(height / 2)))
    if len(tapes) == 0:
        return image
    # Clusterize tapes which labels is 1:
    clusters = clusterize([tapes_plane[i] for i in range(len(tapes_plane)) if labels[i] == 1])
    update_labels(labels, clusters)
    # Build broken line for each cluster:
    lines_plane = []
    banned = []
    for c in np.unique(labels):
        if c != -1:
            group = [i for i in range(len(tapes)) if labels[i] == c]
            line = broken_line([tapes_plane[i] for i in group])
            banned += [list(tapes_plane[group[i]]) for i in line]
            lines_plane.append([list(tapes_plane[group[i]]) for i in line])
    # Extend broken lines with tapes which labels is -1:
    noise = np.array([tapes_plane[i] for i in range(len(tapes_plane)) if labels[i] == -1])
    kd_tree = KDTree(noise, 15)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i, point in enumerate(tapes):
        cv2.circle(image, point, 1, color=(50, 0, 195), thickness=4)
    for i in range(len(lines_plane)):
        lines_plane[i] = extend_line(image, lines_plane[i], kd_tree, noise, banned)

    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for line in lines_plane:
        line = plane_to_image(line, (int(width / 2), int(height / 2)))
        for p, q in zip(line[:-1], line[1:]):
            cv2.line(image, p, q, (210, 195, 0), thickness=2)
            cv2.circle(image, p, 1, color=(195, 0, 50), thickness=4)
            cv2.circle(image, q, 1, color=(195, 0, 50), thickness=4)

    for i, point in enumerate(tapes):
        if labels[i] == -1:
            cv2.circle(image, point, 1, color=(50, 0, 195), thickness=4)
        else:
            # cv2.circle(image, point, 1, color=(195, 0, 50), thickness=4)
            cv2.putText(image, str(tapes_plane[i][0])[:4] + ", " + str(tapes_plane[i][1])[:4], point, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 190, 0))

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

