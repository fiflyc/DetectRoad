import cv2
import numpy as np


def divide_image(image, vlines, hlines):
    """
    Divides image into rectangular parts
    :param image: image for dividing.
    :param vlines: list of x-coordinates of separating vertical lines (including borders of the image).
    :param hlines: list of y-coordinates of separating horizontal lines (including borders of the image).
    :return: 2D list of images.
    """

    result = []
    for up, down in zip(hlines[:-1], hlines[1:]):
        result.append([])
        for left, right in zip(vlines[:-1], vlines[1:]):
            result[-1].append(image[up:down, left:right])
    return result


def unite_images(images):
    """
    Unites 2D list of images.
    :param images: 2D list of images. Images in one row should have similar height, images in one column - similar width.
    :return: united image.
    """

    width = sum([np.shape(image)[1] for image in images[0]])
    height = sum([np.shape(images[i][0])[0] for i in range(np.shape(images)[0])])
    result = [[0 for _ in range(width)] for _ in range(height)]

    H = 0
    for row in images:
        W = 0
        for image in row:
            for i in range(np.shape(image)[0]):
                for j in range(np.shape(image)[1]):
                    result[H + i][W + j] = image[i][j]
            W += np.shape(image)[1]
        H += np.shape(row[0])[0]

    return np.array(result)


def segmented_threshold(image, vlines, hlines, blur_sizes, block_sizes, Cs):
    """
    Applies :func:`~cv2.adaptiveThreshold` for every part of image divided by some horizontal and vertical lines.
    Each part of the image has own parameters of :func:`~cv2.adaptiveThreshold`.
    :param image: input image with one chanel.
    :param vlines: list of x-coordinates of separating vertical lines (including borders of the image).
    :param hlines: list of y-coordinates of separating horizontal lines (including borders of the image).
    :param blur_sizes: blur parameters for each part of image.
    :param block_sizes: 2D list of parameters ``blockSize`` in :func:`~cv2.adaptiveThreshold`.
    :param Cs: 2D list of parameters ``C`` in :func:`~cv2.adaptiveThreshold`.
    :return: stamped image.
    """

    images = divide_image(image, vlines, hlines)
    for i in range(np.shape(images)[0]):
        for j in range(np.shape(images)[1]):
            images[i][j] = cv2.adaptiveThreshold(src=cv2.blur(images[i][j], blur_sizes[i][j]),
                                                 maxValue=255,
                                                 adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 thresholdType=cv2.THRESH_BINARY,
                                                 blockSize=block_sizes[i][j],
                                                 C=Cs[i][j])

    return unite_images(images)


def adaptive_stamp(image):
    """
    Applies :func:`~adaptive_stamp.segmented_threshold` with some parameters to the grayscale image.
    :param image: input image with one channel.
    :return: stamped image.
    """

    H, W = np.shape(image)
    vlines = [0, W]
    hlines = [int(H * p) for p in [0, 0.25, 0.4, 0.6, 1]]
    blure_sizes = [[(20, 20)], [(2, 2)], [(2, 2)], [(10, 10)]]
    block_sizes = [[81], [7], [21], [71]]
    Cs = [[20], [-10], [-5], [-5]]

    return segmented_threshold(image, vlines, hlines, blure_sizes, block_sizes, Cs)