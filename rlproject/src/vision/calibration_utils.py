from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

DEFAULT_IMAGE_SIZE = (2592, 1944)


@dataclass(frozen=True)
class CalibrationData:
    K: np.ndarray
    dist_coeffs: np.ndarray
    new_camera_matrix: np.ndarray
    roi: Tuple[int, int, int, int]


def build_new_camera_matrix(K, dist_coeffs, image_size=DEFAULT_IMAGE_SIZE, alpha=0):
    return cv2.getOptimalNewCameraMatrix(K, dist_coeffs, image_size, alpha, image_size)


def load_calibration_data(path, image_size=DEFAULT_IMAGE_SIZE, alpha=0):
    data = np.load(Path(path))
    K = data["K"]
    dist_coeffs = data["dist_coeffs"]
    new_camera_matrix, roi = build_new_camera_matrix(K, dist_coeffs, image_size=image_size, alpha=alpha)
    return CalibrationData(K=K, dist_coeffs=dist_coeffs, new_camera_matrix=new_camera_matrix, roi=roi)


def save_calibration_data(path, ret, K, dist_coeffs, rvecs, tvecs, new_camera_matrix):
    np.savez(
        Path(path),
        ret=ret,
        K=K,
        dist_coeffs=dist_coeffs,
        rvecs=rvecs,
        tvecs=tvecs,
        new_camera_matrix=new_camera_matrix,
    )
