import numpy as np
from numpy.linalg import norm
from coordinates import ROAD


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

    assert len(paths[longest]) > 1

    return paths[longest]


def can_be_next(point, last, direction):
    if norm(point - last) > ROAD.WIDTH * 0.5:
        return False
    if len(direction) == 0:
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


def extend_line(line, kd_tree, points, banned):
    line = find_longest_extension(line[0], np.array(line[0]) - np.array(line[1]), kd_tree, points, banned) + line
    line = line + find_longest_extension(line[0], np.array(line[0]) - np.array(line[1]), kd_tree, points, banned)
    return line


def find_longest_extension(start, direction, kd_tree, points, banned):
    longest = []
    print(banned)
    for i in kd_tree.query_radius(X=[start], r=ROAD.WIDTH * 0.5)[0]:
        print(i, points[i])
        if i not in banned and can_be_next(np.array(points[i]), np.array(start), direction):
            direction = np.array(points[i]) - np.array(start)
            path = find_longest_extension(points[i], direction, kd_tree, points, banned + [i])
            if len(longest) < len(path) + 1:
                longest = [i] + path

    banned += longest

    return longest
