import traceback
import numpy as np
import cv2
import time
from collections import deque
from matplotlib import pyplot as plt
import pygame
import mediapipe as mp

from stable_baselines3 import SAC
from ultralytics import YOLO

# 导入自定义的库
from custom_env import EnvironmentRenderer, RehabilitationEnv
from camera_calibration.camera_calibration import CameraCalibration
from robot_control.ur_control import URControl  
from cv.hand_detect import HandDetection
from cv.metric import PatientTrajectoryAnalyzer
from cv.get_workspace import get_workspace

# ================= 1. 初始化配置与全局变量 =================
pygame.init()
grid_s = 10
cell_s = 50
w_env, h_env = 15, 10  # 环境的物理宽度和高度 (单位)
screen = pygame.display.set_mode((int(grid_s * cell_s * 1.5), int(grid_s * cell_s)))

trajectory_robot = deque(maxlen=40)
trajectory = deque(maxlen=60)
distance_list = deque(maxlen=1000)

# ================= 2. 实例化各个模块 =================
cv_model = YOLO(r'C:\Users\admin\Desktop\huifeng\rlproject\src\runs\detect\train3\weights\best.onnx')
hand_detector = HandDetection()
cali = CameraCalibration()
analyzer = PatientTrajectoryAnalyzer()

robot_ip = "192.168.1.2"
robot_control = URControl(robot_ip)

env = RehabilitationEnv(training_mode='robot')
env.env_width = w_env
obs, _ = env.reset()
env.arm_blocking_length = 0
env.stride_robot = 0.6
env.stride_hand = 1.0

# 加载 SAC 模型
model_path = r"C:\Users\admin\Desktop\huifeng\RL\src\logs\iterative_training_03_26_13_29\iter_4\robot\best_model.zip"
try:
    model = SAC.load(
        model_path,
        env=env,
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space
        }
    )
except Exception as e:
    print(f"模型加载失败: {e}")
    exit()

# 摄像头初始化
desired_width, desired_height = 2592, 1944
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)  
# cv2.namedWindow('edges', cv2.WINDOW_NORMAL)  

if not cap.isOpened():
    print("Error: 无法打开摄像头")
    exit()

# 先读取一帧，获取画面尺寸(w, h)，用于后续边界限幅
ret, frame = cap.read()
undistorted_frame = cali.undistort_frame(frame)
undistorted_frame = get_workspace(undistorted_frame)
undistorted_frame = cv2.rotate(undistorted_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
h, w = undistorted_frame.shape[:2]

# ================= 3. 运行前变量准备 (核心修复区) =================
pixel_per_cm = 10.0 
last_hand = np.zeros(2, dtype=np.float32)
last_action = [0, 0]
fixed_point = [10, 10]
position_hand_env = [0, 0]
step = 0
frame_count = 0
fps = 0.0
fps_ts = time.time()

render = EnvironmentRenderer(grid_size=10, cell_size=50)

# Matplotlib 非阻塞初始化
# plt.ion()
# fig, ax = plt.subplots(figsize=(5,4))
# bins = np.linspace(0, 10, 15)
# ax.set_title("Distance Distribution (Real-time)")
# ax.set_xlabel("Distance Value")
# ax.set_ylabel("Density")
# plt.show(block=False)

print("系统初始化完成，准备开始闭环控制...")

# ---------------------------------------------------------
# 【终极修复】：分离“肉体”与“灵魂”
# 获取初始真实位置，以此作为“虚拟目标点”的起点
# ---------------------------------------------------------
*init_robot_world, z_init, rx_init, ry_init, rz_init = robot_control.get_robot_pose()
init_robot_pixel = cali.world_to_pixel(init_robot_world)
# virtual_target_pixel 是 AI 心中的绝对坐标，只受 action 影响，不受物理惯性影响！
virtual_target_pixel = np.array(init_robot_pixel, dtype=np.float64)
last_real_robot_world = np.array(init_robot_world[:2])

start_time = time.perf_counter()
last_control_time = time.time() 

# ================= 4. 主循环 =================
try:
    while True:
        w_pixel, h_pixel = int(w_env * 50), int(h_env * 50)
        # --- A. 图像采集与预处理 ---
        ret, frame = cap.read()
        if not ret: break
        
        undistorted_frame = cali.undistort_frame(frame)
        undistorted_frame = get_workspace(undistorted_frame)
        undistorted_frame = cv2.rotate(undistorted_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # --- B. 视觉目标检测 (YOLO & Hand) ---
        results = cv_model.predict(undistorted_frame, conf=0.7, save=False, imgsz=640, verbose=False)
        undistorted_frame, hand_positions = hand_detector.process_frame(undistorted_frame)

        # 提取微机器人坐标 (YOLO)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x2 - x1 > 100: continue
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                trajectory_robot.append([int(cx), int(cy)])
                cv2.rectangle(undistorted_frame, (x1, y1), (x2, y2), (255, 0, 255), 4)
                
                L = (x2 + y2 - x1 - y1) / 2
                pixel_per_cm = L / 2

        if len(trajectory_robot) >= 2:
            for j in range(1, len(trajectory_robot)):
                thickness = int((j / len(trajectory_robot)) * 4) + 1
                cv2.line(undistorted_frame, trajectory_robot[j - 1], trajectory_robot[j], (0, 255, 255), thickness)

        # 提取手部坐标 (MediaPipe)
        if hand_positions:
            position_hand_env = hand_positions[0] / np.array([w/w_env, h/h_env])
            env.hand_position = position_hand_env
            t_current = time.perf_counter() - start_time
            hand_positions_cm = [hand_positions[0][0]/pixel_per_cm, hand_positions[0][1]/pixel_per_cm]
            analyzer.add_point(t_current, *hand_positions_cm)

        # 提取固定点/手肘
        hsv = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2HSV)
        lower_skin, upper_skin = np.array([0, 30, 60], dtype=np.uint8), np.array([20, 150, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        ys, xs = np.where(mask > 0)
        try:
            idx = np.argmax(ys)
            tip = (xs[idx].item(), ys[idx].item())
            cv2.circle(undistorted_frame, tip, 10, (0, 0, 255), -1)
            fixed_point =[tip[0]*w_env/w, h_env]
        except ValueError:
            fixed_point =[10, 10]

        # --- C. 强化学习推理与机械臂控制 ---
        
        # 1. 获取真实机械臂位姿 (用于 Observation)
        *position_robot_world, z_r, rx_r, ry_r, rz_r = robot_control.get_robot_pose()
        
        # 计算真实物理位移 (Debug 用)
        actual_disp_m = np.linalg.norm(np.array(position_robot_world[:2]) - last_real_robot_world)
        last_real_robot_world = np.array(position_robot_world[:2])

        real_robot_pixel = cali.world_to_pixel(position_robot_world)
        position_robot_env = real_robot_pixel[0]*w_env/w, real_robot_pixel[1]*h_env/h
        env.robot_position = position_robot_env

        # 2. 构建 Observation 向量
        robot_obs = np.array([position_robot_env], dtype=np.float32).flatten()
        hand_obs = np.array([position_hand_env], dtype=np.float32).flatten()
        
        hand_move = hand_obs - last_hand
        distance_to_object = np.linalg.norm(robot_obs - hand_obs)
        distance_obs = np.array([distance_to_object], dtype=np.float32)
        
        boundary_obs = np.array([robot_obs[0], w_env-robot_obs[0], robot_obs[1], h_env-robot_obs[1]])
        
        vec_arm = fixed_point - hand_obs
        dist_arm = np.linalg.norm(vec_arm)
        to_shoulder = env.safe_normalize(vec_arm)
        blocking_point = hand_obs + to_shoulder * min(1, dist_arm)
        
        flat_history = np.array(env.hand_history_buffer).flatten()
        if len(flat_history) < env.history_length * 2:
            flat_history = np.zeros(env.history_length * 2)

        obs_array = np.concatenate((
            robot_obs, hand_obs, distance_obs, boundary_obs, 
            np.array([env.stride_robot]), np.array(fixed_point, dtype=np.float32), 
            blocking_point.flatten(), np.zeros(2).flatten(), flat_history
        ))

        # 3. SAC 预测动作
        action, _ = model.predict(obs_array, deterministic=True)
        action_magnitude = np.linalg.norm(action)
        
        env.hand_history_buffer.append(hand_move)
        last_action = action    
        last_hand = hand_obs
        
        # ---------------------------------------------------------
        # 4. 【核心修复】：将动作累加在“虚拟目标点”上！
        # ---------------------------------------------------------
        # 将 env 中的相对 action 转换为 像素级别的位移量
        action_pixel = action * np.array([w/w_env, h/h_env]) * env.stride_robot
        
        # 累加：无论现在真实机器人在哪，AI 心中的目标点坚定地往前推
        virtual_target_pixel += action_pixel
        
        # 限幅：防止目标点飞出屏幕外导致机械臂撞死
        virtual_target_pixel[0] = np.clip(virtual_target_pixel[0], 50, w-100)
        virtual_target_pixel[1] = np.clip(virtual_target_pixel[1], 50, h-100)
        # 🌟 [新增神级防护]：防积分饱和 (防止虚拟点跑太远)
        # diff_vec = virtual_target_pixel - real_robot_pixel
        # dist_diff = np.linalg.norm(diff_vec)
        # max_lead_pixel = np.linalg.norm(np.array([w_pixel/w_env, h_pixel/h_env]) * env.stride_robot) * 3.0
        # if dist_diff > max_lead_pixel:
        #     virtual_target_pixel = real_robot_pixel + (diff_vec / dist_diff) * max_lead_pixel
        
        # 将虚拟目标点转换为真实世界的目标坐标
        target_position_world = cali.pixel_to_world(virtual_target_pixel.astype(int))
        
        # 保持姿态不变 (使用固定的姿态，防止机械臂乱转)
        rx_c, ry_c, rz_c = 0.175, 0.067, 5.22 
        target_pose = [target_position_world[0], target_position_world[1], 0.116, rx_c, ry_c, rz_c]

        # ---------------------------------------------------------
        # 5. 动态计算 dt 并下发伺服指令
        # ---------------------------------------------------------
        now = time.time()
        actual_dt = now - last_control_time
        safe_dt = np.clip(actual_dt, 0.01, 0.5) 
        last_control_time = now
        
        # 发送伺服目标
        robot_control.servo_robot(target_pose, dt=safe_dt)

        trajectory.append(position_robot_env)
        step += 1

        # # --- D. 可视化与终端监控 ---
        # render.render(obs_array[:2], obs_array[2:4], fixed_point, trajectory, blocking_point.flatten())

        # # Debug 打印：对比期望位移与实际位移
        # # 把环境里的 stride_robot (0.5) 换算成物理世界的预期米数，以检查系统是否跑满了步长
        # expected_disp_m = (action_magnitude * env.stride_robot / w_env) * (w / pixel_per_cm) / 100.0 if pixel_per_cm > 0 else 0
        
        # print(f"Step: {step:04d} | dt: {safe_dt:.3f}s | "
        #       f"AI期望位移: {expected_disp_m*1000:.1f}mm | 真实物理位移: {actual_disp_m*1000:.1f}mm")

        # # 画面叠字
        # blocking_point_pixel = blocking_point[0]*w/w_env, blocking_point[1]*h/h_env
        # mean_hand_vel, max_hand_vel = analyzer.get_vel()
        # cv2.putText(undistorted_frame, f"Mean Vel: {mean_hand_vel:.2f} cm/s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        # cv2.circle(undistorted_frame, (int(blocking_point_pixel[0]), int(blocking_point_pixel[1])), 10, (0, 255, 0), -1)

        # --- D. 可视化与终端监控 ---
        render.render(obs_array[:2], obs_array[2:4], fixed_point, trajectory, blocking_point.flatten())

        # Debug 打印：对比期望位移与实际位移
        # 把环境里的 stride_robot 换算成物理世界的预期米数
        expected_disp_m = (action_magnitude * env.stride_robot / w_env) * (w / pixel_per_cm) / 100.0 if pixel_per_cm > 0 else 0
        
        print(f"Step: {step:04d} | dt: {safe_dt:.3f}s | "
              f"AI期望位移: {expected_disp_m*1000:.1f}mm | 真实物理位移: {actual_disp_m*1000:.1f}mm")

        # ========================================================
        # [新增] 画面叠图：可视化“灵魂(虚拟目标)”与“肉体(真实机器人)”
        # ========================================================
        # 1. 画出真实机械臂位置 (绿色实心圆)
        # real_px = (int(real_robot_pixel[0]), int(real_robot_pixel[1]))
        # cv2.circle(undistorted_frame, real_px, 12, (0, 255, 0), -1) 
        
        # # 2. 画出 AI 的虚拟目标点 (紫色空心圈)
        # virt_px = (int(virtual_target_pixel[0]), int(virtual_target_pixel[1]))
        # cv2.circle(undistorted_frame, virt_px, 15, (255, 0, 255), 2)
        # cv2.circle(undistorted_frame, virt_px, 2, (255, 0, 255), -1) # 靶心
        
        # # 3. 画出连接两者的“橡皮筋” (黄色线段)
        # cv2.line(undistorted_frame, real_px, virt_px, (0, 255, 255), 2)
        
        # # 4. 添加文字标注方便区分
        # cv2.putText(undistorted_frame, "Real", (real_px[0]+15, real_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # cv2.putText(undistorted_frame, "AI Target", (virt_px[0]+15, virt_px[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        # ========================================================

        # 画面其他叠字 (手臂固定点等)
        blocking_point_pixel = blocking_point[0]*w/w_env, blocking_point[1]*h/h_env
        mean_hand_vel, max_hand_vel = analyzer.get_vel()
        cv2.putText(undistorted_frame, f"Mean Vel: {mean_hand_vel:.2f} cm/s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.circle(undistorted_frame, (int(blocking_point_pixel[0]), int(blocking_point_pixel[1])), 10, (0, 255, 0), -1)

        # FPS 计算
        frame_count += 1
        if (time.time() - fps_ts) > 1.0:
            fps = frame_count / (time.time() - fps_ts)
            frame_count = 0
            fps_ts = time.time()
        cv2.putText(undistorted_frame, f"FPS: {int(fps):d}", (w - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Frame', undistorted_frame)

        # time.sleep(0.04)

        key = cv2.waitKey(1) 
        if key == ord('q'):
            break

except Exception as e:
    print(f"\n[Error] 运行中发生异常: {e}")
    traceback.print_exc()

finally:
    print("\n正在安全关闭系统...")
    plt.ioff()
    
    try:
        metrics = analyzer.compute_clinical_metrics()
        print("\n======== 康复评估指标 ========")
        print(metrics)
    except:
        pass
    
    # 紧急刹车并断开连接
    if 'robot_control' in locals():
        try:
            robot_control.stop_robot()
            time.sleep(0.5)
            robot_control.disconnect()
        except:
            pass
            
    hand_detector.release()
    cap.release()
    cv2.destroyAllWindows()
    pygame.display.quit()
    pygame.quit()