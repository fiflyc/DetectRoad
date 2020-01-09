import numpy as np
from numpy.linalg import norm
from coordinates import ROAD, TAPE
import cv2
from coordinates import plane_to_image


def broken_line(points):
    if not points:
        return []
    paths = [[] for _ in points]
    for i in range(len(points)):
        paths[i] = [i] + find_longest_path([i], points)

    longest = 0
    for i in range(1, len(points)):
        if len(paths[longest]) < len(paths[i]):
            longest = i

    return paths[longest]


def can_be_next(point, last, direction):
    if norm(point - last) > ROAD.GAP + TAPE.MAX_DIAG:
        return False
    elif norm(point - last) < ROAD.GAP:
        return False
    elif len(direction) == 0:
        return True
    else:
        cos = np.dot(point - last, direction) / (norm(point - last) * norm(direction))
        return ROAD.MIN_COS <= cos


def find_longest_path(current, points):
    last = current[-1]
    longest = []

    for j in range(len(points)):
        if j not in current:
            direction = []
            if len(current) > 1:
                direction = np.array(points[last]) - np.array(points[current[-2]])

            if can_be_next(np.array(points[j]), np.array(points[last]), direction):
                path = find_longest_path(current + [j], points)
                if len(longest) < len(path) + 1:
                    longest = [j] + path

    return longest


def extend_line(image, line, kd_tree, points, banned):
    extend_line.call = 0
    debug = cv2.VideoWriter('./output/debug_line' + str(extend_line.call) + ".mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))

    left = find_longest_extension(image, debug, line[0], np.array(line[0]) - np.array(line[1]), kd_tree, points, banned)
    left.reverse()
    line = left + line
    line = line + find_longest_extension(image, debug, line[-1], np.array(line[-1]) - np.array(line[-2]), kd_tree, points, banned)
    return line


def find_longest_extension(pic, vid, start, direction, kd_tree, points, banned):
    find_longest_extension.calls += 1
    print(find_longest_extension.calls)
    if find_longest_extension.calls > 2400:
        vid.release()
        exit("Max calls reached")

    longest = []
    qq = plane_to_image([start], (320, 240))[0]
    for i in kd_tree.query_radius(X=[start], r=ROAD.WIDTH * 0.5)[0]:
        if points[i].tolist() not in banned and can_be_next(np.array(points[i]), np.array(start), direction):
            pica = np.copy(pic)
            pp = plane_to_image([points[i].tolist()], (320, 240))[0]
            cv2.line(pica, pp, qq, (210, 195, 0), thickness=2)
            cv2.circle(pica, qq, 1, color=(195, 0, 50), thickness=4)
            cv2.circle(pica, pp, 1, color=(0, 190, 0), thickness=4)
            vid.write(pica)
            path = find_longest_extension(pica, vid, points[i].tolist(), np.array(points[i]) - np.array(start), kd_tree, points, banned + [points[i].tolist()])
            if len(longest) < len(path) + 1:
                longest = [points[i].tolist()] + path

    banned += longest

    return longest


find_longest_extension.calls = 0
