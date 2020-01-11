import numpy as np


class ROAD:
    """
    Road parameters.
    """

    WIDTH = 21.75   # 21.75cm,  width
    GAP = 2.5       # 2.5cm,    gap in the yellow dash line
    MIN_COS = 0.75   # cos of max angle between two near tapes


class TAPE:
    """
    Road yellow tape average parameters.
    Width is the shortest edge, diag is the longest diagonal.
    Width is in [MIN_WIDTH, MAX_WIDTH]. Diagonal is in [MIN_DIAG, MAX_DIAG].
    """

    MAX_WIDTH = 2.6  # 2.6cm
    MIN_WIDTH = 0.8  # 0.8cm
    MAX_DIAG = 9.0   # 9.0cm
    MIN_DIAG = 4.0   # 4.0cm


class TRANS:
    """
    Constants for transforming coordinates.
    """

    C = None    # scaling constant (changed by set_horizon function)
    L = 35      # 35cm
    H = 10      # 10cm
    TAN = 0.25  # 1.1cm / 4.4cm
    SIN = np.sin(np.arctan(0.25))


def set_horizon(y):
    """
    Initialises scaling constant TRANS.C.
    :param y: the distance between the horizon and the center of the image.
    """

    TRANS.C = y / TRANS.SIN


def is_in_horizon(y, center):
    """
    Checks case when point is in the horizon.
    :param y: ordinate of the point in the image coordinate system.
    :param center:ordinate of the center of the image.
    :return: true if point is in the horizon.
    """

    return 0 == TRANS.C * TRANS.SIN - center + y


def image_to_plane(points, center):
    """
    Transforms coordinates of points from image coordinate system to plane. If point is in the horizon, skips it.
    :param points: coordinates to transform.
    :param center: center of the image.
    :return: plane coordinates as numpy array.
    """

    xs, ys = zip(*points)
    # Center of the plain coordinate system in the center of the image:
    xs = [x - center[0] for x in xs]
    ys = [center[1] - y for y in ys]
    # Some projective geometry:
    new_ys = [y * (TRANS.H * TRANS.TAN + TRANS.L) / (TRANS.C * TRANS.SIN - y) for y in ys]
    new_xs = [xs[i] * (TRANS.H * TRANS.TAN + TRANS.L + new_ys[i]) / TRANS.C for i in range(len(points))]

    return np.array(list(zip(new_xs, new_ys)))


def plane_to_image(points, center):
    """
    Transforms coordinates of points from plane coordinate system to image.
    :param points: coordinates to transform.
    :param center: center of the image.
    :return: image coordinates as numpy array.
    """

    xs, ys = zip(*points)
    # Some projective geometry:
    new_ys = [y * TRANS.C * TRANS.SIN / (TRANS.H * TRANS.TAN + TRANS.L + y) for y in ys]
    new_xs = [xs[i] * TRANS.C / (TRANS.H * TRANS.TAN + TRANS.L + ys[i]) for i in range(len(points))]
    # Center of the plain coordinate system in the center of the image:
    new_xs = [int(x + center[0]) for x in new_xs]
    new_ys = [int(center[1] - y) for y in new_ys]

    return np.array(list(zip(new_xs, new_ys)))
