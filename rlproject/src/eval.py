"""
M-HECS Evaluation System - 3-Thread Decoupled Architecture
=========================================================
Vision Thread  (~15fps):  cap.read() → undistort → YOLO → MediaPipe → vision_queue
RL Thread      (~30Hz):   obs_build → PPO.predict → action_queue
Control Thread (125Hz):   latest_action + task_logic → servo_robot（主线程）

修改记录:
- [Arch] 3线程解耦：RL推理从控制线程分离，控制频率从~50Hz → 125Hz
- [Arch] HandTracker 加锁保证 Vision/RL 两线程并发安全
- [Arch] 控制层重复上一帧 action（queue 无新数据时沿用 last_action_pixel）
- [Arch] 渲染限速至 ~15Hz，不占控制线程时间
- [Fix]  用 Dead Reckoning 替换卡尔曼滤波
- [Fix]  Queue 清旧帧逻辑修复（get 再 put）
- [Fix]  LeagueGame desired_virtual_target 不再无限累积
- [Fix]  Tracking shape_name 在 task_elapsed>=20 时不再未定义
- [Fix]  undistorted_frame 为 None 时跳过 putText 和渲染
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

_rl_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _rl_root not in sys.path:
    sys.path.insert(0, _rl_root)

from stable_baselines3 import PPO
from ultralytics import YOLO

from custom_env import EnvironmentRenderer
from camera_calibration.camera_calibration import CameraCalibration
from robot_control.ur_control import URControl
from cv.hand_detect import HandDetection
from cv.get_workspace import get_workspace
from callbacks.trajectory_debug_callback import TrajectoryDebugCallback

# ========================================================
# 1. 系统与测评参数配置
# ========================================================
w_env, h_env  = 15, 10
CONTROL_FREQ  = 125          # 控制线程目标频率 Hz
RL_FREQ       = 30           # RL 推理线程目标频率 Hz
RENDER_EVERY  = 8            # 控制线程每 N 帧渲染一次（125/8 ≈ 15Hz）
RX_C, RY_C, RZ_C = 0.193, 0.067, 5.3

STRIDE = 0.3

# EVAL_TASKS = ['Sprint', 'Tracking', 'LeagueGame', 'Boundary']
EVAL_TASKS = ['LeagueGame']
current_task_idx = 0

eval_results = {
    'Sprint':     {'catch_times': [], 'peak_vels': []},
    'Tracking':   {'rmse_list': [], 'jerk_list': []},
    'LeagueGame': {'is_caught': False, 'survival_time': 0.0, 'dist_list': []},
    'Boundary':   {'min_x': 999, 'max_x': 0, 'min_y': 999, 'max_y': 0, 'vel_list': []}
}

# ========================================================
# 2. VisionResult 数据结构
# ========================================================
@dataclass
class VisionResult:
    hand_positions:     list        # [(cx, cy), ...] 像素坐标
    hand_detected:      bool
    robot_trajectory:   list        # [(cx, cy), ...] 机器人轨迹像素坐标
    hand_env:           np.ndarray  # hand 环境坐标 (2,)
    pixel_per_cm:       float
    undistorted_frame:  np.ndarray


# ========================================================
# 3. Dead Reckoning 手部追踪器（线程安全版）
#    Vision Thread 写（update_vision）
#    RL Thread     读（predict / velocity）
#    两处并发 → 加锁
# ========================================================
class HandTracker:
    def __init__(self, w_env=15, h_env=10, vel_alpha=0.6):
        self._lock = threading.Lock()
        self.pos  = np.array([w_env * 3 / 4, h_env / 3], dtype=np.float32)
        self.vel  = np.zeros(2, dtype=np.float32)
        self.last_vision_time = None
        self.w_env     = w_env
        self.h_env     = h_env
        self.vel_alpha = vel_alpha

    def update_vision(self, new_pos: np.ndarray, t_now: float):
        """Vision Thread 调用：直接信任视觉位置，差分估算速度（带低通滤波）。"""
        new_pos = np.array(new_pos, dtype=np.float32)
        with self._lock:
            if self.last_vision_time is not None:
                dt = t_now - self.last_vision_time
                if dt > 1e-3:
                    raw_vel = (new_pos - self.pos) / dt
                    self.vel = self.vel_alpha * self.vel + (1 - self.vel_alpha) * raw_vel
            self.pos = new_pos.copy()
            self.last_vision_time = t_now

    def predict(self, t_now: float) -> np.ndarray:
        """RL Thread / Control Thread 调用：匀速外推当前手部位置。"""
        with self._lock:
            if self.last_vision_time is None:
                return self.pos.copy()
            pos = self.pos.copy()
            vel = self.vel.copy()
            lvt = self.last_vision_time
        # 锁外计算
        dt        = t_now - lvt
        predicted = pos + vel * dt
        predicted[0] = np.clip(predicted[0], 0.3, self.w_env - 0.3)
        predicted[1] = np.clip(predicted[1], 0.3, self.h_env - 0.3)
        return predicted.astype(np.float32)

    @property
    def velocity(self) -> np.ndarray:
        with self._lock:
            return self.vel.copy()


# ========================================================
# 4. Vision Thread
# ========================================================
class VisionThread(threading.Thread):
    def __init__(self, cap, cali, cv_model, w_px, h_px, w_env, h_env, result_queue):
        super().__init__(daemon=True)
        self.cap           = cap
        self.cali          = cali
        self.cv_model      = cv_model
        self.w_px          = w_px
        self.h_px          = h_px
        self.w_env         = w_env
        self.h_env         = h_env
        self.result_queue  = result_queue
        self.hand_detector = HandDetection()
        self.running       = True
        self.frame_count   = 0
        self._fps          = 0.0
        self._last_fps_time = time.perf_counter()

    def run(self):
        print("[VisionThread] 启动")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            undistorted_frame = cv2.rotate(
                get_workspace(self.cali.undistort_frame(frame)),
                cv2.ROTATE_90_COUNTERCLOCKWISE
            )

            # YOLO 检测机器人
            robot_trajectory = []
            pixel_per_cm     = 10.0
            results = self.cv_model.predict(undistorted_frame, conf=0.7,
                                            save=False, imgsz=640, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if x2 - x1 > 100:
                        continue
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    robot_trajectory.append((int(cx), int(cy)))
                    pixel_per_cm = ((x2 - x1) + (y2 - y1)) / 2 / 2

            # MediaPipe 手部检测
            undistorted_frame, hand_positions = self.hand_detector.process_frame(undistorted_frame)

            # 手部像素 → 环境坐标
            hand_env = np.zeros(2, dtype=np.float32)
            if hand_positions:
                hand_env = np.array(hand_positions[0], dtype=np.float32) / np.array(
                    [self.w_px / self.w_env, self.h_px / self.h_env], dtype=np.float32
                )

            # 绘制机器人轨迹
            if len(robot_trajectory) >= 2:
                for j in range(1, len(robot_trajectory)):
                    thickness = int((j / len(robot_trajectory)) * 4) + 1
                    cv2.line(undistorted_frame,
                             robot_trajectory[j - 1], robot_trajectory[j],
                             (0, 255, 255), thickness)

            result = VisionResult(
                hand_positions=hand_positions,
                hand_detected=len(hand_positions) > 0,
                robot_trajectory=robot_trajectory,
                hand_env=hand_env,
                pixel_per_cm=pixel_per_cm,
                undistorted_frame=undistorted_frame
            )

            # 非阻塞写入：队列满时先 get 丢掉旧帧，再 put 新帧
            try:
                self.result_queue.put_nowait(result)
            except queue.Full:
                self.result_queue.get_nowait()
                self.result_queue.put_nowait(result)

            # FPS 统计
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
# 5. RL Inference Thread（新增）
#    职责：obs 构建 + PPO.predict → action_queue
#    不发任何 servo 指令
# ========================================================
class RLInferenceThread(threading.Thread):
    def __init__(self, rl_model, hand_tracker, robot_control, cali,
                 action_queue, w_env, h_env, w_px, h_px):
        super().__init__(daemon=True)
        self.rl_model      = rl_model
        self.hand_tracker  = hand_tracker
        self.robot_control = robot_control
        self.cali          = cali
        self.action_queue  = action_queue
        self.w_env = w_env
        self.h_env = h_env
        self.w_px  = w_px
        self.h_px  = h_px
        self.running = True

        # RL 线程自身状态
        self.last_action         = np.zeros(2, dtype=np.float32)
        self.hand_history_buffer = deque([np.zeros(2)] * 16, maxlen=16)
        self._dt = 1.0 / RL_FREQ   # 约 33ms

    def run(self):
        print("[RLThread] 启动")
        while self.running:
            t0    = time.perf_counter()
            t_now = time.time()

            # 手部位置（Dead Reckoning 预测）
            position_hand_env = self.hand_tracker.predict(t_now)
            vel               = self.hand_tracker.velocity
            self.hand_history_buffer.append(vel * self._dt)

            # 机器人位置（RL 线程自己读，保证 obs 是当前时刻最新值）
            *pos_world, _, _, _, _ = self.robot_control.get_robot_pose()
            robot_pixel        = self.cali.world_to_pixel(pos_world)
            position_robot_env = np.array([
                robot_pixel[0] * self.w_env / self.w_px,
                robot_pixel[1] * self.h_env / self.h_px
            ], dtype=np.float32)

            dist_hand_robot = float(np.linalg.norm(position_robot_env - position_hand_env))
            boundary_obs    = np.array([
                position_robot_env[0], self.w_env - position_robot_env[0],
                position_robot_env[1], self.h_env - position_robot_env[1]
            ], dtype=np.float32)
            flat_history = np.array(self.hand_history_buffer).flatten()

            obs_array = np.concatenate((
                position_robot_env,
                position_hand_env,
                [dist_hand_robot],
                boundary_obs,
                [STRIDE],
                self.last_action,
                flat_history
            )).astype(np.float32)

            action, _ = self.rl_model.predict(obs_array, deterministic=True)
            self.last_action = action.copy()

            print(f"[RLThread] obs={obs_array} | action={action}")

            # 非阻塞写入 action_queue（丢弃旧 action）
            try:
                self.action_queue.put_nowait(action)
            except queue.Full:
                self.action_queue.get_nowait()
                self.action_queue.put_nowait(action)

            # 频率锁定（目标 RL_FREQ Hz）
            elapsed    = time.perf_counter() - t0
            sleep_time = self._dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("[RLThread] 退出")

    def stop(self):
        self.running = False


# ========================================================
# 6. 硬件与模型初始化
# ========================================================
pygame.init()
screen = pygame.display.set_mode((int(10 * 50 * 1.5), int(10 * 50)))

print("[初始化] 加载视觉模型...")
cv_model = YOLO(r'C:\Users\admin\Desktop\huifeng\RL\rlproject\src\runs\detect\train3\weights\best.onnx')
cali     = CameraCalibration()

print("[初始化] 连接机械臂...")
robot_control = URControl("192.168.1.2")

print("[初始化] 加载 RL 决策模型...")
rl_model_path = r"C:\Users\admin\Desktop\huifeng\RL\src\logs\dual_iterative_0509_0945\iteration_10\robot\robot\best_model.zip"
try:
    rl_model = PPO.load(rl_model_path, custom_objects={'learning_rate': 0.0, 'optimizer_class': None})
except Exception as e:
    print(f"PPO 模型加载失败: {e}")
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
# 7. 线程间通信队列 & 共享对象
# ========================================================
vision_queue = queue.Queue(maxsize=1)   # VisionThread → Control Thread
action_queue = queue.Queue(maxsize=1)   # RLThread     → Control Thread

hand_tracker = HandTracker(w_env=w_env, h_env=h_env, vel_alpha=0.6)

# ========================================================
# 8. 启动子线程（顺序：Vision → 等第一帧 → RL → 等第一个action）
# ========================================================
vision_thread = VisionThread(
    cap=cap, cali=cali, cv_model=cv_model,
    w_px=w_px, h_px=h_px, w_env=w_env, h_env=h_env,
    result_queue=vision_queue
)
vision_thread.start()

print("[初始化] 等待第一帧视觉数据...")
first_vision = vision_queue.get(timeout=10.0)
if first_vision.hand_detected:
    hand_tracker.update_vision(first_vision.hand_env, time.time())
vision_queue.put_nowait(first_vision)  # 塞回去，控制线程第一帧可用

rl_thread = RLInferenceThread(
    rl_model=rl_model, hand_tracker=hand_tracker,
    robot_control=robot_control, cali=cali,
    action_queue=action_queue,
    w_env=w_env, h_env=h_env, w_px=w_px, h_px=h_px
)
rl_thread.start()

print("[初始化] 等待第一个 RL action...")
first_action = action_queue.get(timeout=10.0)
action_queue.put_nowait(first_action)  # 塞回去

# ========================================================
# 9. 初始化轨迹记录 Callback
# ========================================================
trajectory_callback = TrajectoryDebugCallback(
    episode_id=time.strftime("%Y%m%d_%H%M%S")
)
trajectory_callback.reset()

# ========================================================
# 10. 辅助函数
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
    target_pose  = [center_world[0], center_world[1], 0.116, RX_C, RY_C, RZ_C]
    robot_control.rtde_c.moveL(target_pose, 0.2, 0.2, asynchronous=False)
    for i in range(3, 0, -1):
        print(f">>> 请受试者准备，{i} 秒后开始...")
        time.sleep(1)
    print(">>> 评估开始！")
    return center_pixel.astype(np.float64)


# 预生成 Task 2 理想轨迹点云
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
}

# ========================================================
# 11. 主控制循环 (125Hz) —— 精简版，只负责 servo 指令
#     RL 推理已移至 RLInferenceThread，控制层重复上一帧 action
# ========================================================
try:
    center_pixel = safe_transition_to_center(EVAL_TASKS[current_task_idx])

    virtual_target_pixel   = center_pixel.copy()
    desired_virtual_target = center_pixel.copy()
    MAX_SAFE_STRIDE = 0.6

    fixed_point    = [10, 10]
    task_start_time   = time.time()
    last_control_time = time.time()
    task_elapsed   = 0.0

    sprint_target_env        = np.array([3.0, 3.0])
    sprint_catch_count       = 0
    sprint_target_spawn_time = time.time()

    # 渲染缓存
    cached_undistorted_frame = None
    cached_robot_trajectory  = []
    cached_pixel_per_cm      = 10.0
    render_counter           = 0

    # ← 控制层核心：记住上一帧 action，queue 无新数据时沿用
    last_action_pixel = np.zeros(2, dtype=np.float32)

    while current_task_idx < len(EVAL_TASKS):
        t_loop_start = time.perf_counter()
        current_task = EVAL_TASKS[current_task_idx]
        t_now        = time.time()
        task_elapsed = t_now - task_start_time

        # -------- A. 非阻塞读取最新视觉结果 --------
        try:
            new_vision = vision_queue.get_nowait()
            if new_vision.hand_detected:
                hand_tracker.update_vision(new_vision.hand_env, t_now)
            cached_undistorted_frame = new_vision.undistorted_frame
            cached_robot_trajectory  = new_vision.robot_trajectory
            cached_pixel_per_cm      = new_vision.pixel_per_cm
        except queue.Empty:
            pass

        # Dead Reckoning：每个控制周期都调用 predict 外推手部位置
        position_hand_env = hand_tracker.predict(t_now)
        inst_vel = float(np.linalg.norm(hand_tracker.velocity))

        # -------- B. 获取机器人位置（用于抓取判断 & 渲染） --------
        *position_robot_world, _, _, _, _ = robot_control.get_robot_pose()
        real_robot_pixel   = cali.world_to_pixel(position_robot_world)
        position_robot_env = np.array([
            real_robot_pixel[0] * w_env / w_px,
            real_robot_pixel[1] * h_env / h_px
        ], dtype=np.float32)
        dist_hand_robot = np.linalg.norm(position_robot_env - position_hand_env)

        # -------- B.1 记录轨迹 --------
        trajectory_callback.record_step(
            robot_pos=position_robot_env.copy(),
            hand_pos=position_hand_env.copy()
        )

        # 固定点检测
        if cached_undistorted_frame is not None:
            hsv  = cv2.cvtColor(cached_undistorted_frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([0, 30, 60]), np.array([20, 150, 255]))
            ys, xs = np.where(mask > 0)
            try:
                fixed_point = [xs[np.argmax(ys)].item() * w_env / w_px, h_env]
            except ValueError:
                pass

        # -------- C. 任务状态机 --------
        task_finished = False

        # --- Sprint ---
        if current_task == 'Sprint':
            desired_virtual_target[0] = sprint_target_env[0] * w_px / w_env
            desired_virtual_target[1] = sprint_target_env[1] * h_px / h_env

            if len(eval_results['Sprint']['peak_vels']) <= sprint_catch_count:
                eval_results['Sprint']['peak_vels'].append(inst_vel)
            else:
                eval_results['Sprint']['peak_vels'][sprint_catch_count] = \
                    max(eval_results['Sprint']['peak_vels'][sprint_catch_count], inst_vel)

            if cached_undistorted_frame is not None:
                cv2.putText(cached_undistorted_frame,
                            f"Task 1: Sprint ({sprint_catch_count}/5)",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

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

        # --- Tracking ---
        elif current_task == 'Tracking':
            time_left = 20 - task_elapsed

            # [Fix] 先确定 shape_name，再做 putText
            if task_elapsed < 10.0:
                shape_name = 'Circle'
                t_c = task_elapsed * 1.2
                target_x = w_env / 2 + (w_env / 3) * math.cos(t_c)
                target_y = h_env / 2 + (h_env / 3) * math.sin(t_c)
            else:
                shape_name = 'Figure-8'
                t_8 = (task_elapsed - 10.0) * 1.2
                target_x = w_env / 2 + (w_env / 3) * math.sin(t_8)
                target_y = h_env / 2 + (w_env / 3) * math.sin(t_8) * math.cos(t_8)

            if cached_undistorted_frame is not None:
                cv2.putText(cached_undistorted_frame,
                            f"Task 2: Tracking [{shape_name}] ({time_left:.1f}s)",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

            desired_virtual_target[0] = target_x * w_px / w_env
            desired_virtual_target[1] = target_y * h_px / h_env

            path_points       = ideal_trajectories[shape_name]
            distances_to_path = np.linalg.norm(path_points - position_hand_env, axis=1)
            cross_track_error = float(np.min(distances_to_path))
            eval_results['Tracking']['rmse_list'].append(cross_track_error)
            eval_results['Tracking']['jerk_list'].append(inst_vel)

            if task_elapsed >= 20.0:
                task_finished = True

        # --- LeagueGame ---
        elif current_task == 'LeagueGame':
            if cached_undistorted_frame is not None:
                cv2.putText(cached_undistorted_frame,
                            f"Task 3: Catching ({30 - task_elapsed:.1f}s)",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

            # 非阻塞读取最新 RL action
            # queue 有新数据 → 更新 last_action_pixel
            # queue 空       → 沿用上一帧 last_action_pixel（重复上一帧）
            try:
                new_action = action_queue.get_nowait()
                last_action_pixel = new_action * np.array([w_px / w_env, h_px / h_env]) * STRIDE
            except queue.Empty:
                pass  # last_action_pixel 保持不变

            # [Fix] 以 virtual_target_pixel 为基准，不无限累积
            desired_virtual_target  = virtual_target_pixel.copy()
            desired_virtual_target += last_action_pixel

            eval_results['LeagueGame']['dist_list'].append(float(dist_hand_robot))

            if dist_hand_robot < 1.5:
                eval_results['LeagueGame']['is_caught']     = True
                eval_results['LeagueGame']['survival_time'] = task_elapsed
                print(f"  -> Robot CAUGHT at {task_elapsed:.2f}s!")
                task_finished = True
            elif task_elapsed >= 30.0:
                eval_results['LeagueGame']['is_caught']     = False
                eval_results['LeagueGame']['survival_time'] = 30.0
                print("  -> Robot survived the full 30s!")
                task_finished = True

        # --- Boundary ---
        elif current_task == 'Boundary':
            if cached_undistorted_frame is not None:
                cv2.putText(cached_undistorted_frame,
                            f"Task 4: ROM ({20 - task_elapsed:.1f}s)",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

            perimeter = 2 * (w_env - 4) + 2 * (h_env - 4)
            progress  = (task_elapsed / 20.0) * perimeter
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
            res['min_x'] = min(res['min_x'], float(position_hand_env[0]))
            res['max_x'] = max(res['max_x'], float(position_hand_env[0]))
            res['min_y'] = min(res['min_y'], float(position_hand_env[1]))
            res['max_y'] = max(res['max_y'], float(position_hand_env[1]))
            res['vel_list'].append(inst_vel)

            if task_elapsed >= 20.0:
                task_finished = True

        # -------- D. 物理执行与安全限幅 --------
        desired_virtual_target[0] = np.clip(desired_virtual_target[0], 50, w_px - 50)
        desired_virtual_target[1] = np.clip(desired_virtual_target[1], 50, h_px - 50)

        max_pixel_step = MAX_SAFE_STRIDE * (w_px / w_env)
        diff_vec   = desired_virtual_target - virtual_target_pixel
        dist_pixel = np.linalg.norm(diff_vec)
        if dist_pixel > max_pixel_step:
            virtual_target_pixel += (diff_vec / dist_pixel) * max_pixel_step
        else:
            virtual_target_pixel = desired_virtual_target.copy()

        target_position_world = cali.pixel_to_world(virtual_target_pixel.astype(int))
        target_pose = [target_position_world[0], target_position_world[1], 0.116, RX_C, RY_C, RZ_C]

        actual_dt = time.time() - last_control_time
        safe_dt   = np.clip(actual_dt, 0.005, 0.1)
        last_control_time = time.time()

        robot_control.servo_robot(target_pose, dt=safe_dt)

        # -------- E. 渲染（限速 ~15Hz，避免 imshow 拖慢控制线程） --------
        render_counter += 1
        if render_counter >= RENDER_EVERY and cached_undistorted_frame is not None:
            render_counter = 0
            frame_to_show  = cached_undistorted_frame.copy()
            cv2.circle(frame_to_show,
                       (int(real_robot_pixel[0]), int(real_robot_pixel[1])), 20, (0, 255, 0), -1)
            cv2.circle(frame_to_show,
                       (int(virtual_target_pixel[0]), int(virtual_target_pixel[1])), 10, (255, 0, 255), 2)
            cv2.imshow('M-HECS Evaluation', frame_to_show)
            if cv2.waitKey(1) == ord('q'):
                break

        # -------- F. 任务切换 --------
        if task_finished:
            current_task_idx += 1
            if current_task_idx < len(EVAL_TASKS):
                center_p = safe_transition_to_center(EVAL_TASKS[current_task_idx])
                virtual_target_pixel   = center_p.copy()
                desired_virtual_target = center_p.copy()
                last_action_pixel      = np.zeros(2, dtype=np.float32)  # 重置 action
                task_start_time        = time.time()

        # -------- G. 125Hz 频率锁定 --------
        elapsed    = time.perf_counter() - t_loop_start
        sleep_time = (1.0 / CONTROL_FREQ) - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except Exception as e:
    print(f"\n[Error] 测评异常: {e}")
    traceback.print_exc()

finally:
    print("\n[系统] 测评结束，正在安全关闭...")

    # 保存轨迹数据
    if trajectory_callback.step_count > 0:
        trajectory_callback.save_episode()

    # 停止顺序：先停 servo → 停 RL → 停 Vision → 断开硬件
    if 'robot_control' in locals():
        robot_control.rtde_c.servoStop()
        time.sleep(0.5)

    rl_thread.stop()
    rl_thread.join(timeout=2.0)

    vision_thread.stop()
    vision_thread.join(timeout=2.0)

    if 'robot_control' in locals():
        robot_control.disconnect()

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()


# ========================================================
# 12. 报告生成
# ========================================================
import matplotlib.gridspec as gridspec


def normalize_score(value, bound_0, bound_100):
    score = 100.0 * (value - bound_0) / (bound_100 - bound_0)
    return max(0.0, min(100.0, score))


def generate_radar_report(results):
    print("\n[系统] 正在生成 M-HECS 评估报告...")

    avg_catch_time = np.mean(results['Sprint']['catch_times']) if results['Sprint']['catch_times'] else 4.0
    score_shooting = normalize_score(avg_catch_time, bound_0=3, bound_100=0.8)

    avg_rmse = np.mean(results['Tracking']['rmse_list']) if results['Tracking']['rmse_list'] else 2.5
    score_tracking = normalize_score(avg_rmse, bound_0=2, bound_100=0.0)

    survival_t    = results['LeagueGame']['survival_time']
    dist_list     = results['LeagueGame']['dist_list']
    avg_game_dist = np.mean(dist_list) if dist_list else 10.0
    time_score_normalized = normalize_score(survival_t, bound_0=10, bound_100=2.0)
    dist_score_normalized = normalize_score(avg_game_dist, bound_0=8.0, bound_100=3)
    score_catching = (time_score_normalized * 0.6) + (dist_score_normalized * 0.4)

    b        = results['Boundary']
    area     = max(0, b['max_x'] - b['min_x']) * max(0, b['max_y'] - b['min_y'])
    max_area = (w_env - 4) * (h_env - 4)
    area_score_normalized = normalize_score(area, bound_0=0.0, bound_100=max_area)
    vel_list  = b['vel_list']
    mean_jerk = np.mean(np.abs(np.diff(vel_list))) if len(vel_list) > 1 else 3.0
    jerk_score_normalized = normalize_score(mean_jerk, bound_0=3.0, bound_100=0.0)
    score_rom = (area_score_normalized * 0.5) + (jerk_score_normalized * 0.5)

    s_shoot = score_shooting / 100.0
    s_track = score_tracking / 100.0
    s_catch = score_catching / 100.0
    s_rom   = score_rom / 100.0

    mhecs_total = 100.0 * (0.20 * s_shoot + 0.30 * s_track + 0.30 * s_catch + 0.20 * s_rom)
    est_fma     = 66.0  * (0.10 * s_shoot + 0.45 * s_track + 0.05 * s_catch + 0.40 * s_rom)
    est_arat    = 57.0  * (0.40 * s_shoot + 0.15 * s_track + 0.35 * s_catch + 0.10 * s_rom)

    labels = ['Shooting\n(Power & Reaction)', 'Tracking\n(Synergy & Smoothness)',
              'Catching\n(Cognitive Interception)', 'ROM\n(Workspace & Stability)']
    stats  = np.array([score_shooting, score_tracking, score_catching, score_rom])
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    stats  = np.concatenate((stats, [stats[0]]))
    angles += angles[:1]

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(11, 6))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[3, 2])

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
                 bbox=dict(boxstyle='round,pad=1', facecolor='#ecf0f1',
                           edgecolor='#bdc3c7', alpha=0.8))

    plt.suptitle('M-HECS: Magnetically-actuated Hand-Eye Coordination Scale',
                 size=16, fontweight='bold', color='#2c3e50', y=0.95)
    plt.tight_layout()
    save_path = 'm_hecs_radar_clinical_report.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 报告已保存: {save_path}")
    plt.show()


generate_radar_report(eval_results)