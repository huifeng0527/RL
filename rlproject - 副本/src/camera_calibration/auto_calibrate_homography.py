import numpy as np
import cv2
import time
from robot_control.ur_control import URControl
from cv.get_workspace import get_workspace
from ultralytics import YOLO

desired_width = 2592
desired_height = 1944

calibration_data = np.load('camera_calibration/calibration_data.npz')
K = calibration_data['K']
dist_coeffs = calibration_data['dist_coeffs']
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K, dist_coeffs, (desired_width, desired_height), 0, (desired_width, desired_height))
model = YOLO("../rl-project/runs/detect/train4/weights/best.pt")
robotip = '192.168.1.2'
ur_control = URControl(robotip)

def get_target_pixel(frame):
    results = model.predict(frame, conf=0.5, save=False,imgsz=frame.shape[1::-1])
    if len(results) == 0 :
        raise ValueError("No objects detected")
    elif len(results[0].boxes) >= 1:
        raise ValueError(f"Multiple objects detected")
    r = results[0]
    boxes = r.boxes
    for box in boxes:
        # 提取边界框坐标
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # 计算中心点
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # 在图像上绘制边界框和中心点
        cv2.rectangle(undistorted_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(undistorted_frame, (cx, cy), 5, (255, 0, 0), -1)
        cv2.putText(undistorted_frame, f'({cx}, {cy})', (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        print(f"Detected object at center: ({cx}, {cy})")
    return [cx, cy]

object_points = []
img_points = []

num_samples = 12  # 采集点数量
dx_range = 0.1  # x方向移动范围
dy_range = 0.1  # y方向移动范围
center_point = ur_control.get_robot_pose()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

for i in range(num_samples):
    pose = ur_control.get_robot_pose()
    dx = np.random.uniform(-dx_range, dx_range)
    dy = np.random.uniform(-dy_range, dy_range)
    new_pose = [pose[0] + dx, pose[1] + dy, pose[2], pose[3], pose[4], pose[5]]

    ur_control.move_robot(new_pose)
    time.sleep(2)  # 等待机械臂到位

    # 读取当前摄像头图像
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        continue

    # 畸变矫正
    undistorted_frame = cv2.undistort(frame, K, dist_coeffs, None, new_camera_matrix)
    undistorted_frame = get_workspace(undistorted_frame)

    # 获取目标在图像中的像素坐标
    pixel = get_target_pixel(undistorted_frame)
    img_points.append([pixel[0], pixel[1]])

    # 获取当前机械臂末端世界坐标
    object_points.append([new_pose[0], new_pose[1]])

    ur_control.move_robot(center_point)
    time.sleep(2)
    print(f"Sample {i+1}: World {object_points[-1]}, Pixel {img_points[-1]}")
    cv2.imshow("Undistorted Frame", undistorted_frame)
    cv2.waitKey(500)


cap.release()

# 计算单应性矩阵
object_points_np = np.array(object_points, dtype=np.float32)
img_points_np = np.array(img_points, dtype=np.float32)
H, status = cv2.findHomography(object_points_np, img_points_np)
np.save('Homography_matrix.npy', H)
# print("Homography matrix saved.")
