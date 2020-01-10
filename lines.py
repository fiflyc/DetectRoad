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

    return paths[longest]


def can_be_next(point, last, direction):
    point = np.array(point)
    last = np.array(last)
    if norm(point - last) > ROAD.WIDTH * 0.5:
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

            if can_be_next(points[j], points[last], direction):
                path = find_longest_path(current + [j], points)
                if len(longest) < len(path) + 1:
                    longest = [j] + path

    return longest


def extend_line(line, kd_tree, points, banned):
    left = find_longest_extension(line[0], np.array(line[0]) - np.array(line[1]), kd_tree, points, banned)
    left.reverse()
    line = left + line
    line = line + find_longest_extension(line[-1], np.array(line[-1]) - np.array(line[-2]), kd_tree, points, banned)
    return line


def find_longest_extension(start, direction, kd_tree, points, banned):
    longest = []

    for i in kd_tree.query_radius(X=[start], r=ROAD.WIDTH * 0.5)[0]:
        if points[i].tolist() not in banned and can_be_next(points[i], start, direction):
            path = find_longest_extension(points[i].tolist(), np.array(points[i]) - np.array(start), kd_tree, points,
                                          banned + [points[i].tolist()])
            if len(longest) < len(path) + 1:
                longest = [points[i].tolist()] + path

    banned += longest

    return longest


def concatenate_lines(first, second):
    if can_be_next(second[0], first[-1], np.array(first[-1]) - np.array(first[-2])) and \
            can_be_next(first[-1], second[0], np.array(second[0]) - np.array(second[1])):
        return first + second
    elif can_be_next(second[-1], first[-1], np.array(first[-1]) - np.array(first[-2])) and \
            can_be_next(first[-1], second[-1], np.array(second[-1]) - np.array(second[-2])):
        second.reverse()
        return first + second
    elif can_be_next(first[0], second[-1], np.array(second[-1]) - np.array(second[-2])) and \
            can_be_next(second[-1], first[0], np.array(first[0]) - np.array(first[1])):
        return second + first
    elif can_be_next(first[-1], second[-1], np.array(second[-1]) - np.array(second[-2])) and \
            can_be_next(second[-1], first[-1], np.array(first[-1]) - np.array(first[-2])):
        second.reverse()
        return second + first
    else:
        return None


def pair_up(lines):
    result = []
    for i in range(len(lines)):
        was_united = False
        for j in range(i + 1, len(lines)):
            line = concatenate_lines(lines[i], lines[j])
            if line is not None:
                result.append(line)
                was_united = True
                break

        if not was_united:
            result.append(lines[i])

    return result


def unite_lines(lines):
    current = pair_up(lines)
    while len(current) < len(lines):
        lines = current
        current = pair_up(lines)
    return current


def line_score(line, wight, height):
    B = 0
    D = height
    for point in line:
        B += min(point[0], wight - point[0]) ** 2
        D = min(D, height - point[1])
    B /= len(line)

    return B / D
