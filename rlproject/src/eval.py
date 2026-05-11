"""
M-HECS Evaluation System - 2-Thread Decoupled Architecture
=========================================================
Vision Thread (~10-15fps): cap.read() → undistort → YOLO → MediaPipe → queue
Control Thread (25Hz):      queue.get_nowait() → PPO → servo_robot → render
"""
import traceback
import numpy as np
import cv2
import time
import os
import sys
import math
import threading
import queue
from collections import deque
from matplotlib import pyplot as plt
import pygame
from dataclasses import dataclass

# Add RL root to sys.path so PPO can unpickle src.utils.feature_extractors
_rl_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _rl_root not in sys.path:
    sys.path.insert(0, _rl_root)

from stable_baselines3 import PPO
from ultralytics import YOLO

# ========================================================
# 导入自定义库
# ========================================================
from custom_env import EnvironmentRenderer
from camera_calibration.camera_calibration import CameraCalibration
from robot_control.ur_control import URControl
from cv.hand_detect import HandDetection
from cv.get_workspace import get_workspace

# ========================================================
# 1. 系统与测评参数配置
# ========================================================
w_env, h_env = 15, 10  # 环境物理尺寸 (单位)
CONTROL_FREQ = 25.0
RX_C, RY_C, RZ_C = 0.193, 0.067, 5.3  # 机械臂固定姿态

# 评估任务配置 (状态机)
EVAL_TASKS = ['Sprint', 'Tracking', 'LeagueGame', 'Boundary']
current_task_idx = 0

# 记录评估数据的字典
eval_results = {
    'Sprint': {'catch_times': [], 'peak_vels': []},
    'Tracking': {'rmse_list': [], 'jerk_list': []},
    'LeagueGame': {'is_caught': False, 'survival_time': 0.0, 'dist_list': []},
    'Boundary': {'min_x': 999, 'max_x': 0, 'min_y': 999, 'max_y': 0, 'vel_list': []}
}

# ========================================================
# 2. Vision Result 数据结构 (跨线程传递)
# ========================================================
@dataclass
class VisionResult:
    """Vision Thread 产出的数据结构，Control Thread 消费"""
    hand_positions: list  # [(cx, cy), ...] 像素坐标
    robot_trajectory: list  # [(cx, cy), ...] 机器人轨迹像素坐标
    hand_env: np.ndarray  # hand 环境坐标 (2,)
    pixel_per_cm: float
    undistorted_frame: np.ndarray  # 用于渲染的帧


# ========================================================
# 3. Vision Thread (自由跑 ~10-15fps)
# ========================================================
class VisionThread(threading.Thread):
    """
    视觉线程：独立于控制频率运行
    - 每次循环读取新帧，运行 YOLO + MediaPipe
    - 结果写入 queue (maxsize=1)，队列满时丢弃旧结果
    - 线程内部创建 HandDetection，规避跨线程 MediaPipe 问题
    """
    def __init__(self, cap: cv2.VideoCapture, cali, cv_model,
                 w_px, h_px, w_env, h_env,
                 result_queue: queue.Queue):
        super().__init__(daemon=True)
        self.cap = cap
        self.cali = cali
        self.cv_model = cv_model
        self.w_px = w_px
        self.h_px = h_px
        self.w_env = w_env
        self.h_env = h_env
        self.result_queue = result_queue

        # ⚠️ HandDetection 必须在 VisionThread 内部创建，禁止跨线程共享
        self.hand_detector = HandDetection()

        self.running = True
        self.frame_count = 0
        self._fps = 0.0
        self._last_fps_time = time.perf_counter()

    def run(self):
        print("[VisionThread] 启动")
        while self.running:
            t0 = time.perf_counter()

            ret, frame = self.cap.read()
            if not ret:
                continue

            undistorted_frame = cv2.rotate(
                get_workspace(self.cali.undistort_frame(frame)),
                cv2.ROTATE_90_COUNTERCLOCKWISE
            )

            # YOLO 检测机器人
            robot_trajectory = []
            pixel_per_cm = 10.0
            results = self.cv_model.predict(undistorted_frame, conf=0.7,
                                             save=False, imgsz=640, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if x2 - x1 > 100:
                        continue
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    robot_trajectory.append((int(cx), int(cy)))
                    pixel_per_cm = ((x2 + y2 - x1 - y1) / 2) / 2

            # MediaPipe 手部检测 (在 VisionThread 内部创建和使用，安全)
            undistorted_frame, hand_positions = self.hand_detector.process_frame(undistorted_frame)

            # 手部位置 → 环境坐标
            hand_env = np.zeros(2, dtype=np.float32)
            if hand_positions:
                hand_env = np.array(hand_positions[0]) / np.array([self.w_px / self.w_env,
                                                                     self.h_px / self.h_env],
                                                                    dtype=np.float32)

            # 绘制机器人轨迹
            if len(robot_trajectory) >= 2:
                for j in range(1, len(robot_trajectory)):
                    thickness = int((j / len(robot_trajectory)) * 4) + 1
                    cv2.line(undistorted_frame,
                             robot_trajectory[j - 1], robot_trajectory[j],
                             (0, 255, 255), thickness)

            result = VisionResult(
                hand_positions=hand_positions,
                robot_trajectory=robot_trajectory,
                hand_env=hand_env,
                pixel_per_cm=pixel_per_cm,
                undistorted_frame=undistorted_frame
            )

            # 非阻塞写入，队列满时丢弃旧帧（最新帧优先）
            try:
                self.result_queue.put_nowait(result)
            except queue.Full:
                try:
                    self.result_queue.put_nowait(result)  # 再试一次（清空旧帧）
                except queue.Full:
                    pass

            # FPS 计算
            self.frame_count += 1
            t1 = time.perf_counter()
            if t1 - self._last_fps_time >= 5.0:
                self._fps = self.frame_count / (t1 - self._last_fps_time)
                self.frame_count = 0
                self._last_fps_time = t1
                print(f"[VisionThread] ~{self._fps:.1f} fps")

        self.hand_detector.release()
        print("[VisionThread] 退出")

    def stop(self):
        self.running = False


# ========================================================
# 4. 硬件与模型初始化
# ========================================================
pygame.init()
screen = pygame.display.set_mode((int(10 * 50 * 1.5), int(10 * 50)))

print("[初始化] 加载视觉模型...")
cv_model = YOLO(r'C:\Users\admin\Desktop\huifeng\rlproject\src\runs\detect\train3\weights\best.onnx')
cali = CameraCalibration()

print("[初始化] 连接机械臂...")
robot_control = URControl("192.168.1.2")

print("[初始化] 加载 RL 决策大模型...")
rl_model_path = r"C:\Users\admin\Desktop\huifeng\RL\src\logs\ablation_study_0409_0922\2_MLP_LSTM\best_model.zip"
try:
    rl_model = PPO.load(rl_model_path, custom_objects={'learning_rate': 0.0, 'optimizer_class': None})
except Exception as e:
    print(f"PPO 模型加载失败, 请检查路径: {e}")
    exit()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2592)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1944)
cv2.namedWindow('M-HECS Evaluation', cv2.WINDOW_NORMAL)

if not cap.isOpened():
    exit()

ret, frame = cap.read()
undistorted_frame = get_workspace(cali.undistort_frame(frame))
undistorted_frame = cv2.rotate(undistorted_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
h_px, w_px = undistorted_frame.shape[:2]

render = EnvironmentRenderer(grid_size=10, cell_size=50)

# ========================================================
# 5. 启动 Vision Thread
# ========================================================
vision_queue = queue.Queue(maxsize=1)  # 丢弃旧帧，只保留最新
vision_thread = VisionThread(
    cap=cap,
    cali=cali,
    cv_model=cv_model,
    w_px=w_px,
    h_px=h_px,
    w_env=w_env,
    h_env=h_env,
    result_queue=vision_queue
)
vision_thread.start()

# ========================================================
# 6. 辅助函数
# ========================================================
def safe_normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-8:
        return np.zeros_like(v)
    return v / norm


def safe_transition_to_center(task_name):
    print(f"\n{'=' * 40}")
    print(f"即将开始评估任务: 【{task_name}】")
    print(f"{'=' * 40}")

    robot_control.rtde_c.servoStop()
    center_pixel = np.array([w_px / 2, h_px / 2])
    center_world = cali.pixel_to_world(center_pixel.astype(int))
    target_pose = [center_world[0], center_world[1], 0.116, RX_C, RY_C, RZ_C]

    robot_control.rtde_c.moveL(target_pose, 0.2, 0.2, asynchronous=False)

    for i in range(3, 0, -1):
        print(f">>> 请受试者准备，{i} 秒后开始...")
        time.sleep(1)
    print(">>> 评估开始！")
    return center_pixel.astype(np.float64)


# 预生成 Task 2 的理想轨迹点云
t_vals = np.linspace(0, 2 * math.pi, 500)
ideal_trajectories = {
    'Circle': np.column_stack((
        w_env / 2 + (w_env / 3) * np.cos(t_vals),
        h_env / 2 + (h_env / 3) * np.sin(t_vals)
    )),
    'Figure-8': np.column_stack((
        w_env / 2 + (w_env / 3) * np.sin(t_vals),
        h_env / 2 + (w_env / 3) * np.sin(t_vals) * np.cos(t_vals)
    )),
    'Line': np.column_stack((
        w_env / 2 + (w_env / 3) * np.sin(t_vals),
        np.full_like(t_vals, h_env / 2)
    ))
}

# ========================================================
# 7. 主控制循环 (25Hz 定时)
# ========================================================
try:
    center_pixel = safe_transition_to_center(EVAL_TASKS[current_task_idx])

    virtual_target_pixel = center_pixel.copy()
    desired_virtual_target = center_pixel.copy()
    MAX_SAFE_STRIDE = 0.6

    hand_history_buffer = deque([np.zeros(2)] * 16, maxlen=16)
    last_hand_env = np.zeros(2)
    fixed_point = [10, 10]

    task_start_time = time.time()
    last_control_time = time.time()
    task_elapsed = 0.0

    sprint_target_env = np.array([3.0, 3.0])
    sprint_catch_count = 0
    sprint_target_spawn_time = time.time()

    # 缓存最近一次视觉结果（graceful degradation）
    cached_vision: VisionResult = None

    while current_task_idx < len(EVAL_TASKS):
        t_loop_start = time.perf_counter()
        current_task = EVAL_TASKS[current_task_idx]

        # -------- A. 非阻塞读取最新视觉结果 --------
        try:
            cached_vision = vision_queue.get_nowait()
        except queue.Empty:
            pass  # 用上一帧缓存，控制频率不被视觉绑架

        t_now = time.time()
        task_elapsed = t_now - task_start_time

        # -------- B. 使用缓存的视觉数据 --------
        if cached_vision is not None:
            hand_positions = cached_vision.hand_positions
            robot_trajectory = cached_vision.robot_trajectory
            pixel_per_cm = cached_vision.pixel_per_cm
            undistorted_frame = cached_vision.undistorted_frame

            position_hand_env = cached_vision.hand_env
            inst_vel = 0.0
            dt_vision = max(t_now - last_control_time, 0.01)

            if len(hand_positions) > 0:
                hand_move = position_hand_env - last_hand_env
                inst_vel = np.linalg.norm(hand_move) / dt_vision
                hand_history_buffer.append(hand_move)
                last_hand_env = position_hand_env.copy()

        # -------- C. 获取机器人位置 --------
        *position_robot_world, _, _, _, _ = robot_control.get_robot_pose()
        real_robot_pixel = cali.world_to_pixel(position_robot_world)
        position_robot_env = np.array([real_robot_pixel[0] * w_env / w_px,
                                        real_robot_pixel[1] * h_env / h_px],
                                       dtype=np.float32)
        dist_hand_robot = np.linalg.norm(position_robot_env - position_hand_env)

        # 固定点检测
        if cached_vision is not None:
            hsv = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([20, 150, 255]))
            ys, xs = np.where(mask > 0)
            try:
                fixed_point = [xs[np.argmax(ys)].item() * w_env / w_px, h_env]
            except ValueError:
                pass

        # -------- D. 任务状态机逻辑 --------
        task_finished = False

        # Sprint
        if current_task == 'Sprint':
            cv2.putText(undistorted_frame, f"Task 1: ({sprint_catch_count}/5)",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

            desired_virtual_target[0] = sprint_target_env[0] * w_px / w_env
            desired_virtual_target[1] = sprint_target_env[1] * h_px / h_env

            if len(eval_results['Sprint']['peak_vels']) <= sprint_catch_count:
                eval_results['Sprint']['peak_vels'].append(inst_vel)
            else:
                eval_results['Sprint']['peak_vels'][sprint_catch_count] = \
                    max(eval_results['Sprint']['peak_vels'][sprint_catch_count], inst_vel)

            dist_to_target = np.linalg.norm(sprint_target_env - position_hand_env)
            if dist_to_target < 1.5:
                catch_time = t_now - sprint_target_spawn_time
                eval_results['Sprint']['catch_times'].append(catch_time)
                print(f"  -> Target {sprint_catch_count + 1} caught in {catch_time:.2f}s!")
                sprint_catch_count += 1
                if sprint_catch_count >= 5:
                    task_finished = True
                else:
                    min_travel_distance = 3.0
                    sprint_target_env = np.random.uniform(2.0, [w_env - 2.0, h_env - 2.0])
                    while np.linalg.norm(sprint_target_env - position_hand_env) < min_travel_distance:
                        sprint_target_env = np.random.uniform(2.0, [w_env - 2.0, h_env - 2.0])
                    sprint_target_spawn_time = t_now

        # Tracking
        elif current_task == 'Tracking':
            time_left = 20 - task_elapsed
            t = task_elapsed * 1.2

            if task_elapsed < 10.0:
                shape_name = 'Circle'
                target_x = w_env / 2 + (w_env / 3) * math.cos(t)
                target_y = h_env / 2 + (h_env / 3) * math.sin(t)
            elif task_elapsed < 20.0:
                shape_name = 'Figure-8'
                t_8 = (task_elapsed - 10.0) * 1.2
                target_x = w_env / 2 + (w_env / 3) * math.sin(t_8)
                target_y = h_env / 2 + (h_env / 3) * np.sin(t_8) * np.cos(t_8)

            cv2.putText(undistorted_frame, f"Task 2: Tracking [{shape_name}] ({time_left:.1f}s)",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

            desired_virtual_target[0] = target_x * w_px / w_env
            desired_virtual_target[1] = target_y * h_px / h_env

            path_points = ideal_trajectories[shape_name]
            distances_to_path = np.linalg.norm(path_points - position_hand_env, axis=1)
            cross_track_error = np.min(distances_to_path)

            eval_results['Tracking']['rmse_list'].append(cross_track_error)
            eval_results['Tracking']['jerk_list'].append(inst_vel)

            if task_elapsed >= 20:
                task_finished = True

        # LeagueGame
        elif current_task == 'LeagueGame':
            cv2.putText(undistorted_frame, f"Task 3: catching ({30 - task_elapsed:.1f}s)",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

            robot_obs = position_robot_env
            hand_obs = position_hand_env
            distance_obs = np.array([dist_hand_robot], dtype=np.float32)
            boundary_obs = np.array([robot_obs[0], w_env - robot_obs[0],
                                     robot_obs[1], h_env - robot_obs[1]], dtype=np.float32)
            flat_history = np.array(hand_history_buffer).flatten()

            vec_arm = np.array(fixed_point) - hand_obs
            dist_arm = np.linalg.norm(vec_arm)
            to_shoulder = safe_normalize(vec_arm)
            blocking_point = hand_obs + to_shoulder * min(1, dist_arm)

            obs_array = np.concatenate((
                robot_obs, hand_obs, distance_obs, boundary_obs,
                np.array([0.6]), flat_history
            )).astype(np.float32)

            action, _ = rl_model.predict(obs_array, deterministic=True)
            action_pixel = action * np.array([w_px / w_env, h_px / h_env]) * 0.6
            desired_virtual_target += action_pixel

            eval_results['LeagueGame']['dist_list'].append(dist_hand_robot)

            if dist_hand_robot < 1.5:
                eval_results['LeagueGame']['is_caught'] = True
                eval_results['LeagueGame']['survival_time'] = task_elapsed
                print(f"  -> Robot CAUGHT by patient at {task_elapsed:.2f}s!")
                task_finished = True
            elif task_elapsed >= 30.0:
                eval_results['LeagueGame']['is_caught'] = False
                eval_results['LeagueGame']['survival_time'] = 30.0
                print(f"  -> Robot survived the full 30s!")
                task_finished = True

        # Boundary
        elif current_task == 'Boundary':
            cv2.putText(undistorted_frame, f"Task 4: ROM ({20 - task_elapsed:.1f}s)",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

            perimeter = 2 * (w_env - 4) + 2 * (h_env - 4)
            progress = (task_elapsed / 20.0) * perimeter
            if progress < w_env - 4:
                tx, ty = 1 + progress, 1
            elif progress < w_env - 4 + h_env - 4:
                tx, ty = w_env - 1, 1 + (progress - (w_env - 4))
            elif progress < 2 * (w_env - 4) + h_env - 4:
                tx, ty = w_env - 1 - (progress - (w_env - 4) - (h_env - 4)), h_env - 1
            else:
                tx, ty = 1, h_env - 1 - (progress - 2 * (w_env - 4) - (h_env - 4))

            desired_virtual_target[0] = tx * w_px / w_env
            desired_virtual_target[1] = ty * h_px / h_env

            res = eval_results['Boundary']
            res['min_x'] = min(res['min_x'], position_hand_env[0])
            res['max_x'] = max(res['max_x'], position_hand_env[0])
            res['min_y'] = min(res['min_y'], position_hand_env[1])
            res['max_y'] = max(res['max_y'], position_hand_env[1])
            res['vel_list'].append(inst_vel)

            if task_elapsed >= 20.0:
                task_finished = True

        # -------- E. 物理执行与安全限幅 --------
        desired_virtual_target[0] = np.clip(desired_virtual_target[0], 50, w_px - 50)
        desired_virtual_target[1] = np.clip(desired_virtual_target[1], 50, h_px - 50)

        max_pixel_step = MAX_SAFE_STRIDE * (w_px / w_env)
        diff_vec = desired_virtual_target - virtual_target_pixel
        dist_pixel = np.linalg.norm(diff_vec)

        if dist_pixel > max_pixel_step:
            virtual_target_pixel += (diff_vec / dist_pixel) * max_pixel_step
        else:
            virtual_target_pixel = desired_virtual_target.copy()

        target_position_world = cali.pixel_to_world(virtual_target_pixel.astype(int))
        target_pose = [target_position_world[0], target_position_world[1], 0.116, RX_C, RY_C, RZ_C]

        actual_dt = time.time() - last_control_time
        safe_dt = np.clip(actual_dt, 0.01, 0.2)
        last_control_time = time.time()

        robot_control.servo_robot(target_pose, dt=safe_dt)

        # -------- F. 渲染 --------
        cv2.circle(undistorted_frame,
                   (int(real_robot_pixel[0]), int(real_robot_pixel[1])), 20, (0, 255, 0), -1)
        cv2.circle(undistorted_frame,
                   (int(virtual_target_pixel[0]), int(virtual_target_pixel[1])), 10, (255, 0, 255), 2)
        cv2.imshow('M-HECS Evaluation', undistorted_frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        if task_finished:
            current_task_idx += 1
            if current_task_idx < len(EVAL_TASKS):
                center_p = safe_transition_to_center(EVAL_TASKS[current_task_idx])
                virtual_target_pixel = center_p.copy()
                desired_virtual_target = center_p.copy()
                task_start_time = time.time()

        # -------- G. 25Hz 频率锁定 --------
        elapsed = time.perf_counter() - t_loop_start
        sleep_time = (1.0 / CONTROL_FREQ) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except Exception as e:
    print(f"\n[Error] 测评异常: {e}")
    traceback.print_exc()

finally:
    print("\n[系统] 测评结束，正在安全关闭...")
    vision_thread.stop()
    vision_thread.join(timeout=2.0)
    if 'robot_control' in locals():
        robot_control.rtde_c.servoStop()
        time.sleep(0.5)
        robot_control.disconnect()
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


# ========================================================
# 8. 报告生成
# ========================================================
import matplotlib.gridspec as gridspec


def normalize_score(value, bound_0, bound_100):
    score = 100.0 * (value - bound_0) / (bound_100 - bound_0)
    return max(0.0, min(100.0, score))


def generate_radar_report(results):
    print("\n[系统] 正在基于临床边界生成 M-HECS 评估报告...")

    avg_catch_time = np.mean(results['Sprint']['catch_times']) if results['Sprint']['catch_times'] else 4.0
    score_shooting = normalize_score(avg_catch_time, bound_0=3, bound_100=0.8)

    avg_rmse = np.mean(results['Tracking']['rmse_list']) if results['Tracking']['rmse_list'] else 2.5
    score_tracking = normalize_score(avg_rmse, bound_0=2, bound_100=0.0)

    survival_t = results['LeagueGame']['survival_time']
    dist_list = results['LeagueGame']['dist_list']
    avg_game_dist = np.mean(dist_list) if dist_list else 10.0

    time_score_normalized = normalize_score(survival_t, bound_0=10, bound_100=2.0)
    dist_score_normalized = normalize_score(avg_game_dist, bound_0=8.0, bound_100=3)
    score_catching = (time_score_normalized * 0.6) + (dist_score_normalized * 0.4)

    b = results['Boundary']
    area = max(0, b['max_x'] - b['min_x']) * max(0, b['max_y'] - b['min_y'])
    max_area = (w_env - 4) * (h_env - 4)

    area_score_normalized = normalize_score(area, bound_0=0.0, bound_100=max_area)

    vel_list = b['vel_list']
    mean_jerk = np.mean(np.abs(np.diff(vel_list))) if len(vel_list) > 1 else 3.0
    jerk_score_normalized = normalize_score(mean_jerk, bound_0=3.0, bound_100=0.0)

    score_rom = (area_score_normalized * 0.5) + (jerk_score_normalized * 0.5)

    s_shoot = score_shooting / 100.0
    s_track = score_tracking / 100.0
    s_catch = score_catching / 100.0
    s_rom = score_rom / 100.0

    mhecs_total = 100.0 * (0.20 * s_shoot + 0.30 * s_track + 0.30 * s_catch + 0.20 * s_rom)
    est_fma = 66.0 * (0.10 * s_shoot + 0.45 * s_track + 0.05 * s_catch + 0.40 * s_rom)
    est_arat = 57.0 * (0.40 * s_shoot + 0.15 * s_track + 0.35 * s_catch + 0.10 * s_rom)

    labels = ['Shooting\n(Power & Reaction)', 'Tracking\n(Synergy & Smoothness)',
              'Catching\n(Cognitive Interception)', 'ROM\n(Workspace & Stability)']
    stats = np.array([score_shooting, score_tracking, score_catching, score_rom])

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    stats = np.concatenate((stats, [stats[0]]))
    angles += angles[:1]

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(11, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2])

    ax = plt.subplot(gs[0], polar=True)
    ax.set_facecolor('#f8f9fa')
    ax.grid(color='#bdc3c7', linewidth=1.0, linestyle='--')
    ax.spines['polar'].set_visible(False)
    ax.plot(angles, stats, color='#2980b9', linewidth=2.5, linestyle='solid', marker='o', markersize=6)
    ax.fill(angles, stats, color='#3498db', alpha=0.3)
    ax.plot(angles, [60] * len(angles), color='#e74c3c', linewidth=1.5, linestyle=':')

    ax.set_yticklabels([])
    ax.set_ylim(0, 100)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold', color='#2c3e50')

    ax_text = plt.subplot(gs[1])
    ax_text.axis('off')

    report_text = (
        f"M-HECS Digital Report\n"
        f"{'-' * 25}\n\n"
        f"[ Sub-Task Scores ]\n"
        f"1. Shooting:   {score_shooting:5.1f} / 100\n"
        f"2. Tracking:   {score_tracking:5.1f} / 100\n"
        f"3. Catching:   {score_catching:5.1f} / 100\n"
        f"4. ROM:        {score_rom:5.1f} / 100\n\n"
        f"{'-' * 25}\n"
        f"OVERALL:       {mhecs_total:5.1f} / 100\n\n\n"
        f"[ Clinical Estimation ]\n"
        f"Est. FMA-UE:   {est_fma:5.1f} / 66\n"
        f"(Motor Synergy & Coord.)\n\n"
        f"Est. ARAT:     {est_arat:5.1f} / 57\n"
        f"(Handling & Reaching)"
    )

    ax_text.text(0.0, 0.9, report_text, fontsize=12, family='monospace',
                 verticalalignment='top', color='#2c3e50',
                 bbox=dict(boxstyle='round,pad=1', facecolor='#ecf0f1', edgecolor='#bdc3c7', alpha=0.8))

    plt.suptitle('M-HECS: Magnetically-actuated Hand-Eye Coordination Scale',
                 size=16, fontweight='bold', color='#2c3e50', y=0.95)
    plt.tight_layout()

    save_path = 'm_hecs_radar_clinical_report.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 临床级测评报告已保存为: {save_path}")
    plt.show()


generate_radar_report(eval_results)
