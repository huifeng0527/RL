"""
相机标定模块
用于处理相机畸变矫正和坐标转换
"""
import cv2
import numpy as np
import os


class CameraCalibration:
    """相机标定类"""
    
    def __init__(self, calibration_matrix_path=None, homography_matrix_path=None):
        """
        初始化相机标定
        
        Args:
            calibration_matrix_path: 相机标定矩阵文件路径
            homography_matrix_path: 单应性矩阵文件路径
        """
        # 默认路径（相对于deployment/vision目录）
        if calibration_matrix_path is None:
            calibration_matrix_path = os.path.join(
                os.path.dirname(__file__), 
                'calibration_data.npz'
            )
        if homography_matrix_path is None:
            homography_matrix_path = os.path.join(
                os.path.dirname(__file__), 
                'Homography_matrix.npy'
            )
        
        # 加载单应性矩阵
        if os.path.exists(homography_matrix_path):
            self.H = np.load(homography_matrix_path)
        else:
            raise FileNotFoundError(f"单应性矩阵文件不存在: {homography_matrix_path}")
        
        # 加载相机标定数据
        if os.path.exists(calibration_matrix_path):
            calibration_data = np.load(calibration_matrix_path)
            self.K = calibration_data['K']
            self.dist_coeffs = calibration_data['dist_coeffs']
            self.new_camera_matrix, self.roi = cv2.getOptimalNewCameraMatrix(
                self.K, 
                self.dist_coeffs, 
                (2592, 1944), 
                0, 
                (2592, 1944)
            )
        else:
            raise FileNotFoundError(f"相机标定文件不存在: {calibration_matrix_path}")

    def undistort_frame(self, frame):
        """
        使用相机矩阵进行畸变矫正
        
        Args:
            frame: 输入图像帧
            
        Returns:
            np.ndarray: 矫正后的图像
        """
        undistorted_frame = cv2.undistort(
            frame, 
            self.K, 
            self.dist_coeffs, 
            None, 
            self.new_camera_matrix
        )
        return undistorted_frame

    def pixel_to_world(self, pixel_coords):
        """
        将像素坐标转换为世界坐标系坐标
        
        Args:
            pixel_coords: 像素坐标 [x, y]
            
        Returns:
            np.ndarray: 世界坐标 [x, y]
        """
        p_world = np.linalg.inv(self.H) @ np.array(
            [pixel_coords[0], pixel_coords[1], 1], 
            dtype=np.float32
        )
        p_world /= p_world[2]  # 归一化
        return p_world[:2]

    def world_to_pixel(self, world_coords):
        """
        将世界坐标系坐标转换为像素坐标
        
        Args:
            world_coords: 世界坐标 [x, y]
            
        Returns:
            np.ndarray: 像素坐标 [x, y]
        """
        p_pixel = self.H @ np.array(
            [world_coords[0], world_coords[1], 1], 
            dtype=np.float32
        )
        p_pixel /= p_pixel[2]  # 归一化
        return p_pixel[:2]

    def get_camera_matrix(self):
        """
        获取相机矩阵
        
        Returns:
            tuple: (K, dist_coeffs) 相机内参矩阵和畸变系数
        """
        return self.K, self.dist_coeffs

