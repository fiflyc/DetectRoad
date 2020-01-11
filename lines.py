import numpy as np
from numpy.linalg import norm
from coordinates import ROAD


def broken_line(points):
    """
    Finds longest broken line with short edges and big angles.
    :param points: possible vertexes of broken line.
    :return: broken line as list.
    """

    if len(points) == 0:
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
    """
    Checks that point can be next in the broken line.
    :param point: point to check.
    :param last: last point in the broken line.
    :param direction: vector of last edge of broken line as numpy array.
    :return: True if check passed.
    """

    if norm(point - last) > ROAD.WIDTH * 0.5:
        return False
    elif len(direction) == 0:
        return True
    else:
        cos = np.dot(point - last, direction) / (norm(point - last) * norm(direction))
        return ROAD.MIN_COS <= cos


def find_longest_path(current, points):
    """
    Finds longest extension of broken line with short edges and big angles.
    :param current: broken line as list.
    :param points: possible vertexes of broken line.
    :return: longest extension as list.
    """

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
    """
    Finds longest extension of broken line with short edges and big angles.
    :param line: broken line to extend.
    :param kd_tree: KDTree of possible vertexes of extension.
    :param points: possible vertexes of extension.
    :param banned: points, what can't be in the result.
    :return: extended broken line.
    """

    left = find_longest_extension(line[0], line[0] - line[1], kd_tree, points, banned)
    left.reverse()
    right = find_longest_extension(line[-1], line[-1] - line[-2], kd_tree, points, banned)
    return np.array(left + line.tolist() + right)


def find_longest_extension(start, direction, kd_tree, points, banned):
    longest = []

    for i in kd_tree.query_radius(X=[start], r=ROAD.WIDTH * 0.5)[0]:
        if points[i].tolist() not in banned and can_be_next(points[i], start, direction):
            path = find_longest_extension(points[i], points[i] - start, kd_tree, points, banned + [points[i].tolist()])
            if len(longest) < len(path) + 1:
                longest = [points[i].tolist()] + path

    banned += longest

    return longest


def concatenate_lines(first, second):
    if can_be_next(second[0], first[-1], first[-1] - first[-2]) and can_be_next(first[-1], second[0], second[0] - second[1]):
        return np.append(first, second, axis=0)
    elif can_be_next(second[-1], first[-1], first[-1] - first[-2]) and can_be_next(first[-1], second[-1], second[-1] - second[-2]):
        return np.append(first, np.flip(second, axis=0), axis=0)
    elif can_be_next(first[0], second[-1], second[-1] - second[-2]) and can_be_next(second[-1], first[0], first[0] - first[1]):
        return np.append(second, first, axis=0)
    elif can_be_next(first[-1], second[-1], second[-1] - second[-2]) and can_be_next(second[-1], first[-1], first[-1] - first[-2]):
        return np.append(second, np.flip(first, axis=0), axis=0)
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
    L = np.linalg.norm(line[-1] - line[0])
    for point in line:
        B += min(point[0], wight - point[0]) ** 2
        D = min(D, height - point[1])
    B /= len(line)

    return B / D * L
