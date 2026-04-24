import mediapipe as mp
import cv2
import numpy as np

class HandDetection:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # 初始化 Mediapipe 手部检测模型
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils  # 用于绘制手部关键点

    def process_frame(self, frame):
        """
        处理一帧图像并返回手掌中心位置
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
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                # cv2.circle(frame, (cx, cy), 8, (0, 0, 255), -1)  # 绘制手掌中心点
        return frame, hand_positions

    def release(self):
        """
        释放资源
        """
        self.hands.close()

# 使用示例
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    hand_detector = HandDetection()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        frame, hand_positions = hand_detector.process_frame(frame)

        # 显示处理后的图像
        cv2.imshow('Hand Detection', frame)

        # 按下 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    hand_detector.release()
    cap.release()
    cv2.destroyAllWindows()
