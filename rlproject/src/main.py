import traceback
from stable_baselines3 import PPO,SAC
import numpy as np
import random
import cv2
import time
import mediapipe as mp
from custom_env import CustomEnv,EnvironmentRenderer
import pygame
import rtde_receive
import rtde_control
from cv.get_workspace import get_workspace
from collections import deque
from matplotlib import pyplot as plt

trajectory_robot = deque(maxlen=60)
trajectory = deque(maxlen=60)
# 用 deque 存最近的 distance 值
distance_list = deque(maxlen=1000)

from camera_calibration.camera_calibration import CameraCalibration
from robot_control.ur_control import URControl  
from cv.hand_detect import HandDetection
from cv.hand_move import get_hand_move

from ultralytics import YOLO
cv_model = YOLO(r'C:\Users\admin\Desktop\huifeng\RL\rlproject\src\runs\detect\train3\weights\best.onnx')
hand_detector = HandDetection()
cali = CameraCalibration()
robot_ip = "192.168.1.2"

robot_control = URControl(robot_ip)


env = CustomEnv()
env.random = False

w_env, h_env = 15, 10  # 环境的宽度和高度
# C:\Users\admin\Desktop\huifeng\RL\rlproject\src\model\model_500step.zip
# C:\Users\admin\Desktop\huifeng\RL\src\logs\best_model_sac88\best_model.zip
model = SAC.load(
    r"C:\Users\admin\Desktop\huifeng\RL\rlproject\src\model1\model_500step.zip",
    env=env,
    custom_objects={
        "observation_space": env.observation_space,
        "action_space": env.action_space
    }
)

env.stride_robot=2


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

desired_width = 2592 
desired_height = 1944

cap = cv2.VideoCapture(0)
cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)  # 创建一个窗口来显示矫正后的图像
cv2.namedWindow('edges', cv2.WINDOW_NORMAL)  
if not cap.isOpened():
    print("Error: Could not open camera for demonstration.")
    exit()

clicked_x, clicked_y = 1000, 1000
position_hand_env = [0,0]
object_points = [] # 用于存储世界坐标系中的点
img_points = []  # 用于存储图像坐标系中的点

def mouse_callback(event, x, y, flags, param):
    global clicked_x, clicked_y
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_x, clicked_y = x, y
        return clicked_x, clicked_y

def metric(trajectory):
    """
    你的评估函数占位（保持原样）
    """
    # 这里只返回一个示例分数，按需改
    if len(trajectory) == 0:
        return 0
    return 0.0

cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

frame_count = 0
fps = 0.0
fps_interval = 1.0
fps_ts = time.time()

last_trigger_time = time.time()
freq = 2

last_action = [0,0]

recorded_data = []
last_hand = 0
render =  EnvironmentRenderer(grid_size=10, cell_size=50)

# ---------------- matplotlib 初始化（非阻塞模式） ----------------
plt.ion()
fig, ax = plt.subplots(figsize=(5,4))
# bins 范围按你要显示的 distance 调整，下面只是例子
bins = np.linspace(0, 10, 15)  # 假设距离在 0~300 像素或单位内
ax.set_title("Distance Distribution (实时更新)")
ax.set_xlabel("Distance Value")
ax.set_ylabel("Frequency")
plt.show(block=False)
# ----------------------------------------------------------------
step = 0
terminated = False
terminated_step = 0
fixed_point = [10,10]
try:

    while True:

        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera for demonstration.")
            break
        # 使用 cv2.undistort 对每一帧进行畸变矫正
        undistorted_frame = cali.undistort_frame(frame)
        undistorted_frame = get_workspace(undistorted_frame)
        undistorted_frame = cv2.rotate(undistorted_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_gray = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2GRAY)

        h , w = undistorted_frame.shape[:2]
        # print(h,w)


        results = cv_model.predict(undistorted_frame, conf=0.7, save=False,imgsz=640,verbose=False)

        undistorted_frame, hand_positions = hand_detector.process_frame(undistorted_frame)
        for i, r in enumerate(results):
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x2-x1 > 100:
                    continue
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                trajectory_robot.append([int(cx), int(cy)])
                cv2.rectangle(undistorted_frame, (x1, y1), (x2, y2), (255, 0, 255), 6)

        if len(trajectory_robot) >= 2:
            i = 2
            for j in range(1, len(trajectory_robot)):
                cv2.line(undistorted_frame, trajectory_robot[j - 1], trajectory_robot[j], (0, 255, 255), int(i//2))
                i+=0.2

        # if hand_positions:
        #     position_hand_env = hand_positions[0]/np.array([w/w_env,h/h_env])
        hsv = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2HSV)
        

        lower_skin = np.array([0, 30, 60], dtype=np.uint8)
        upper_skin = np.array([20, 150, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # ys, xs = np.where(mask > 0)

        # try:
        #     idx = np.argmax(ys)
        #     tip = (xs[idx].item(), ys[idx].item())
        #     cv2.circle(frame,tip, 10, (0, 0, 255), -1)
        #     fixed_point = [tip[0]*w_env/w,h_env]

        # except:
        #     fixed_point = [10,10]

        key = cv2.waitKey(1) 
        if key== ord('q'):
            break

        now = time.time()

        if now - last_trigger_time > 1/freq:

            # 获取机器人当前位姿（解包）
            *position_robot_world,z,rx,ry,rz = robot_control.get_robot_pose()

            position_robot_pixel = cali.world_to_pixel(position_robot_world)
            # position_robot_env =  position_robot_pixel[1]*w_env / h, h_env - position_robot_pixel[0]*h_env /w 
            position_robot_pixel = cx,cy
            position_robot_env =  position_robot_pixel[0]*w_env/w, position_robot_pixel[1]*h_env/h

            if not terminated:
                position_hand_env = get_hand_move([np.array(position_hand_env,dtype=np.float32),np.array(position_robot_env,dtype=np.float32),env.stride_robot*0.5])
            else:
                terminated_step += 1
                if terminated_step > 4:
                    fixed_point = [random.randint(0,w_env),h_env]
                    terminated = False
                    terminated_step = 0
            position_hand_pixel = position_hand_env[0]*w/w_env, position_hand_env[1]*h/h_env
            
            robot = np.array([position_robot_env],dtype=np.float32)
            hand = np.array([position_hand_env],dtype=np.float32)
            stride_hand = np.linalg.norm(hand - last_hand)
            print("stride_hand:",stride_hand)
            distance_to_object = np.linalg.norm(robot - hand)
            distance = np.array([distance_to_object],dtype=np.float32)
            boundary = np.array([min(position_robot_env[0],position_robot_env[1],w_env-position_robot_env[0],h_env-position_robot_env[1])],dtype=np.float32)
            last_action = np.array([last_action],dtype=np.float32)
            dist_arm = env.dist_point_to_segment_correct(robot.flatten(),hand.flatten(),[15,10])[0]

            stride_robot = env.stride_robot
            obs = np.concatenate((robot.flatten(),hand.flatten(),last_action.flatten(),distance.flatten(),boundary.flatten(),np.array([dist_arm],dtype=np.float32),np.array(fixed_point,dtype=np.float32),np.array([stride_robot]),np.array([stride_hand]),np.array([w_env,h_env])))

            action, _states = model.predict(obs, deterministic=True)
            last_action = action    
            print(f"obs:{obs}\n action:{action}")
            action_pixel = action * np.array([w/w_env,h/h_env])*stride_robot


            # cv2.circle(undistorted_frame, (int(position_robot_pixel[0]), int(position_robot_pixel[1])), 10, (0, 255, 0), -1)
            # cv2.circle(undistorted_frame, (int(position_hand_pixel[0]), int(position_hand_pixel[1])), 10, (255, 0, 0), -1)

            rx,ry,rz = 0.085,-0.027,4.637

            position_robot_pixel += np.array([action_pixel[0],action_pixel[1]])
            position_robot_world = cali.pixel_to_world(position_robot_pixel)
            robot_control.move_robot([position_robot_world[0],position_robot_world[1],0.125,rx,ry,rz],1/freq)

            step += 1

            last_hand = hand
            last_trigger_time = now
            trajectory.append(position_robot_env)

            # --- 更新 env stride（你原来的逻辑） ---
            # 注意 obs[6] 可能是 numpy array 或标量，强制转 float
            # 但要防止除 0
            obs6 = float(obs[6]) if np.ndim(obs[6]) == 0 or np.ndim(obs[6])==1 else float(np.array(obs[6]).flatten()[0])

            # 避免除零
            obs6 = max(obs6, 1e-6)

            # 指数调整
            base_stride = 2.5
            coef = np.exp(5-obs6)  # 观察值越大，coef 越小
            # env.stride_robot = base_stride * coef

            # 可加上最大值限制
            # env.stride_robot = min(base_stride, env.stride_robot)


            # render 你的环境画面（保持）
            render.render(obs[:2], obs[2:4], fixed_point, trajectory)

            # ---------- 更新直方图：传入一个标量值 ----------
            # 把 obs6 添加到 deque 并更新直方图
            if float(obs6) < 2:
                terminated = True
            distance_list.append(float(obs6))
            # 重绘 hist（非阻塞）
            ax.clear()
            ax.hist(list(distance_list), bins=bins, color='skyblue', alpha=0.7,density=True)
            ax.set_title("Distance Distribution")
            ax.set_xlabel("density")
            ax.set_ylabel("Dis")
            # 自动缩放 y 轴
            ax.relim()
            ax.autoscale_view(True, True, True)
            plt.draw()
            plt.pause(0.001)



        cv2.circle(undistorted_frame, (int(position_robot_pixel[0]), int(position_robot_pixel[1])), 10, (0, 255, 0), -1)
        cv2.circle(undistorted_frame, (int(position_hand_pixel[0]), int(position_hand_pixel[1])), 10, (255, 0, 0), -1)

        cv2.setMouseCallback("Frame", mouse_callback)
        frame_count += 1

        if (time.time() - fps_ts) > fps_interval:
            fps = frame_count / (time.time() - fps_ts)
            frame_count = 0
            fps_ts = time.time()
        # undistorted_frame = cv2.rotate(undistorted_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.putText(undistorted_frame, f"FPS: {int(fps):d}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Frame', undistorted_frame)
        cv2.imshow('edges', mask)

        if step>=1000:
            plt.ioff()  # 关闭交互模式（防止窗口继续刷新）
            # plt.savefig("distance_distribution_2.png", dpi=300, bbox_inches='tight')
            plt.show()  # 重新显示最终静态图
            break
    
    print(sum(distance_list)/len(distance_list))
    robot_control.disconnect()
    hand_detector.release()
    cap.release()
    cv2.destroyAllWindows()
    pygame.display.quit()
    pygame.quit()

except Exception as e:
    print(f"Error occurred: {e}")
    traceback.print_exc()

finally:
    robot_control.disconnect()
    hand_detector.release()
    cap.release()
    cv2.destroyAllWindows()
    pygame.display.quit()
    pygame.quit()
