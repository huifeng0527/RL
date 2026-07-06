from numpy import array

from vision.preprocessing import crop_workspace


def get_workspace(img: array):
    return crop_workspace(img)
