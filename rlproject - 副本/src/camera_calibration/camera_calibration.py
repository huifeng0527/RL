# camera_calibration.py

import cv2
import numpy as np

class CameraCalibration:
    def __init__(self, calibration_matrix_path='./camera_calibration/calibration_data.npz', homography_matrix_path=r'C:\Users\admin\Desktop\huifeng\RL\rlproject\src\camera_calibration\Homography_matrix.npy'):
        self.H = np.load(homography_matrix_path)
        calibration_data = np.load(calibration_matrix_path)
        self.K = calibration_data['K']
        self.dist_coeffs = calibration_data['dist_coeffs']
        self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(self.K, self.dist_coeffs, (2592, 1944), 0, (2592, 1944))

    def undistort_frame(self, frame):
        """ 使用相机矩阵进行畸变矫正 """
        undistorted_frame = cv2.undistort(frame, self.K, self.dist_coeffs, None, self.new_camera_matrix)
        return undistorted_frame

    def pixel_to_world(self, pixel_coords):
        """ 将像素坐标转换为世界坐标系坐标 """
        p_world = np.linalg.inv(self.H) @ np.array([pixel_coords[0], pixel_coords[1], 1], dtype=np.float32)
        p_world /= p_world[2]  # 归一化
        return p_world[:2]

    def world_to_pixel(self, world_coords):
        """ 将世界坐标系坐标转换为像素坐标 """
        p_pixel = self.H @ np.array([world_coords[0], world_coords[1], 1], dtype=np.float32)
        p_pixel /= p_pixel[2]  # 归一化
        return p_pixel[:2]

    def get_camera_matrix(self):
        return self.K, self.dist_coeffs
