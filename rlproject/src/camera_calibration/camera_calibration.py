from pathlib import Path

from vision.calibration_utils import DEFAULT_IMAGE_SIZE, load_calibration_data
from vision.geometry import load_homography, pixel_to_world, world_to_pixel
from vision.paths import DEFAULT_CALIBRATION_DATA_PATH, DEFAULT_HOMOGRAPHY_PATH
from vision.preprocessing import undistort_frame


class CameraCalibration:
    def __init__(
        self,
        calibration_matrix_path=None,
        homography_matrix_path=None,
        image_size=DEFAULT_IMAGE_SIZE,
    ):
        calibration_matrix_path = Path(calibration_matrix_path or DEFAULT_CALIBRATION_DATA_PATH)
        homography_matrix_path = Path(homography_matrix_path or DEFAULT_HOMOGRAPHY_PATH)
        self.H = load_homography(homography_matrix_path)
        calibration_data = load_calibration_data(calibration_matrix_path, image_size=image_size, alpha=0)
        self.K = calibration_data.K
        self.dist_coeffs = calibration_data.dist_coeffs
        self.new_camera_matrix = calibration_data.new_camera_matrix
        self.roi = calibration_data.roi

    def undistort_frame(self, frame):
        return undistort_frame(frame, self.K, self.dist_coeffs, self.new_camera_matrix)

    def pixel_to_world(self, pixel_coords):
        return pixel_to_world(self.H, pixel_coords)

    def world_to_pixel(self, world_coords):
        return world_to_pixel(self.H, world_coords)

    def get_camera_matrix(self):
        return self.K, self.dist_coeffs
