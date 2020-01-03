import cv2
from numpy import shape, array
from numpy.linalg import norm
from coordinates import TAPE, image_to_plane, is_in_horizon
from adaptive_stamp import adaptive_stamp


def check_tape(contour):
    """
    Checks, that contour of the tape has correct length and width.
    :param contour: list of pairs. Be aware - coordinates should be in the plain coordinate system!
    :return: true if contour matches with TAPE parameters.
    """

    if len(contour) == 4:
        k = len(contour)
        diag = max([max([norm(array(contour[i]) - array(contour[j])) for j in range(k) if i != j]) for i in range(k)])
        width = min([min([norm(array(contour[i]) - array(contour[j])) for j in range(k) if i != j]) for i in range(k)])
        if not TAPE.MIN_DIAG * 0.75 <= diag <= TAPE.MAX_DIAG * 1.2:
            return False
        if not TAPE.MIN_WIDTH * 0.75 <= width <= TAPE.MAX_WIDTH * 1.2:
            return False
    elif len(contour) == 2:
        if not TAPE.MIN_DIAG * 0.75 <= norm(array(contour[0]) - array(contour[1])) < TAPE.MAX_DIAG * 1.2:
            return False
    else:
        return False

    return True


def filter_contours(contours, image_shape):
    result = []
    levels = []
    N, M = image_shape

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.1 * cv2.arcLength(contour, closed=True), closed=True)

        if 2 <= len(approx) <= 4:
            center = (sum(approx) / approx.shape[0])[0]
            if is_in_horizon(int(center[1]), int(N / 2)):
                continue
            approx = image_to_plane([(p[0][0], p[0][1]) for p in approx], (int(M / 2), int(N / 2)))

            result.append((int(center[0]), int(center[1])))
            if check_tape(approx):
                levels.append(1)
            else:
                levels.append(-1)

    return result, levels


def label(contour):
    # return "(" + str(center[0])[:5] + ", " + str(center[1])[:5] + ")"
    if len(contour) > 2:
        k = len(contour)
        diag = max([max([norm(array(contour[i]) - array(contour[j])) for j in range(k) if i != j]) for i in range(k)])
        width = min([min([norm(array(contour[i]) - array(contour[j])) for j in range(k) if i != j]) for i in range(k)])
        return str(diag)[:4] + "D, " + str(width)[:4] + "W"
    elif len(contour) == 2:
        return str(norm(array(contour[0]) - array(contour[1])))[:4] + "D"
    else:
        return ""


def detect_tapes(image):
    stamped = adaptive_stamp(image)
    _, contours, _ = cv2.findContours(stamped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return filter_contours(contours, shape(stamped))
