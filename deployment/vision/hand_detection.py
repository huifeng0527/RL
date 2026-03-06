"""
手部检测模块
使用MediaPipe进行手部检测
"""
import mediapipe as mp
import cv2
import numpy as np


class HandDetection:
    """手部检测类"""
    
    def __init__(self, max_num_hands=1, min_detection_confidence=0.5, 
                 min_tracking_confidence=0.5):
        """
        初始化手部检测
        
        Args:
            max_num_hands: 最大检测手数
            min_detection_confidence: 最小检测置信度
            min_tracking_confidence: 最小跟踪置信度
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame):
        """
        处理一帧图像并返回手掌中心位置
        
        Args:
            frame: 输入图像帧
            
        Returns:
            tuple: (处理后的图像, 手部位置列表)
        """
        # 将BGR图像转为RGB格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        hand_positions = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 获取所有关键点的像素坐标
                h, w, _ = frame.shape
                pts = [(int(pt.x * w), int(pt.y * h)) for pt in hand_landmarks.landmark]

                # 计算手掌中心，使用所有关键点的平均值
                cx = int(np.mean([p[0] for p in pts]))
                cy = int(np.mean([p[1] for p in pts]))

                hand_positions.append((cx, cy))

                # 绘制手部关键点和连接线
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
        return frame, hand_positions

    def release(self):
        """释放资源"""
        self.hands.close()

