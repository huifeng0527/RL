"""
摄像头流处理模块
"""
import cv2


class CameraStream:
    """摄像头流类"""
    
    def __init__(self, width=640, height=480, camera_id=0):
        """
        初始化摄像头流
        
        Args:
            width: 图像宽度
            height: 图像高度
            camera_id: 摄像头ID
        """
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self):
        """
        读取一帧图像
        
        Returns:
            np.ndarray: 图像帧
            
        Raises:
            Exception: 如果读取失败
        """
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to read from camera")
        return frame

    def release(self):
        """释放摄像头资源"""
        self.cap.release()

