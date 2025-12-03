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
cv_model = YOLO(r'C:\Users\admin\Desktop\huifeng\rlproject\src\runs\detect\train3\weights\best.onnx')
hand_detector = HandDetection()
cali = CameraCalibration()
robot_ip = "192.168.1.2"

robot_control = URControl(robot_ip)





import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import json
import time
from collections import deque

# --- Constants & Colors ---
COLORS = {
    'bg': (250, 250, 250),
    'grid': (220, 220, 220),
    'robot': (231, 76, 60),      # Red
    'hand': (46, 204, 113),      # Green
    'arm_safe': (200, 200, 200), # Grey
    'arm_block': (230, 126, 34), # Orange
    'trajectory': (52, 152, 219),# Blue
    'text': (50, 60, 80)
}

# --- Biomechanical Filter Class (恢复并启用) ---
class BiomechanicalFilter:
    """模拟不同病理特征的动作滤波器"""
    def __init__(self, mode='healthy', dt=0.02):
        self.mode = mode
        self.dt = dt
        self.t = 0.0
        
        # 帕金森参数
        self.tremor_amp = 0.5
        self.tremor_freq = 5.0 # Hz
        
        # 中风/迟缓参数
        self.lag_buffer = deque(maxlen=5) # 延迟缓冲
        self.speed_factor = 1.0
        
        if mode == 'parkinson':
            self.tremor_amp = 0.8
        elif mode == 'stroke':
            self.speed_factor = 0.5 # 肌无力
            self.lag_buffer = deque(maxlen=10) # 严重延迟

    def apply(self, action, current_pos=None):
        self.t += self.dt
        
        # 1. 基础处理
        final_action = action.copy()
        
        # 2. 病理特征叠加
        if self.mode == 'parkinson':
            # 正弦震颤
            tremor = self.tremor_amp * np.sin(2 * np.pi * self.tremor_freq * self.t)
            noise = np.random.normal(0, 0.1, 2)
            final_action += (noise + tremor) * 0.1 # 震颤叠加
            
        elif self.mode == 'stroke':
            # 延迟 + 减速
            self.lag_buffer.append(action)
            if len(self.lag_buffer) == self.lag_buffer.maxlen:
                final_action = self.lag_buffer[0] * self.speed_factor
            else:
                final_action = np.zeros(2) # 缓冲没满不动
        
        return final_action

# --- Main Environment Class ---
class RehabilitationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, 
                 training_mode='robot', # 'robot' or 'hand'
                 robot_model=None,      # 训练 Hand 时需要的 Robot 对手
                 hand_model=None,       # [新增] 训练 Robot 时需要的 Hand 对手
                 hand_type='healthy',   # 'healthy', 'parkinson', 'stroke'
                 render_mode=None):
        
        super().__init__()
        self.training_mode = training_mode
        self.robot_model = robot_model
        self.hand_model = hand_model    # [新增] 保存手掌模型
        self.hand_type = hand_type
        self.render_mode = render_mode
        
        # --- Dimensions ---
        self.grid_size = 10
        self.cell_size = 50 
        self.env_width = self.grid_size * 1.5
        self.env_height = self.grid_size
        self.margin = 0.3
        
        # --- Physics & Arm ---
        self.max_length_arm = self.grid_size * 0.9
        self.fixed_point = np.array([self.env_width / 2, self.env_height])
        self.arm_blocking_length = 2.0
        
        # --- Filters ---
        self.bio_filter = BiomechanicalFilter(self.hand_type)
        
        # --- Thresholds ---
        self.distance_threshold_collision = 1.0
        self.distance_threshold_penalty = 3.0
        
        # --- Rewards Config ---
        self.reward_hand_catch = 100
        self.reward_robot_caught = -100
        self.reward_arm_hit = -50
        self.reward_bound = -50
        self.reward_step = -0.1 if training_mode == 'hand' else 0.1
        self.reward_survival = 50
        
        # Biomechanical Cost Weights
        self.w_sweep = 10.0
        self.w_effort = 2.0
        
        # --- Movement Params ---
        self.stride_robot_random = [0.8, 1.2]
        self.stride_hand_random = [0.5, 1.0]
        self.hand_move_epsilon = 0.1
        
        self.max_steps = 200
        self.steps = 0
        
        # --- Spaces ---
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observation space definition
        self.history_length = 8
        # obs dim calculation: 
        # robot(2) + hand(2) + history(20) + dist(1) + bounds(4) + stride(1) + fix(2) + block(2) = 34
        self.obs_dim = 30
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # --- Internals ---
        self.robot_position = np.zeros(2)
        self.hand_position = np.zeros(2)
        self.blocking_point = np.zeros(2)
        self.hand_history_buffer = deque(maxlen=self.history_length)
        self.trajectory_points = []
        
        self.window = None
        self.clock = None
        self.random_noise = True
        self.noise_sigma = 0.05

    def safe_normalize(self, v):
        norm = np.linalg.norm(v)
        if norm < 1e-8: return np.zeros_like(v)
        return v / norm

    def _check_line_intersection(self, p1, p2, p3, p4):
        """判断线段相交"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def _calculate_fix_point(self):
        """简单反向运动学"""
        vec = self.hand_position - self.fixed_point
        dist = np.linalg.norm(vec)
        if dist > self.max_length_arm:
            dx = abs(self.hand_position[0] - self.fixed_point[0])
            dy = abs(self.hand_position[1] - self.fixed_point[1])
            max_dx = np.sqrt(max(0, self.max_length_arm**2 - dy**2))
            target_x = self.hand_position[0] - max_dx if self.hand_position[0] < self.fixed_point[0] else self.hand_position[0] + max_dx
            self.fixed_point[0] = 0.8 * self.fixed_point[0] + 0.2 * target_x
            self.fixed_point[0] = np.clip(self.fixed_point[0], 0, self.env_width)


    def _calculate_biomechanical_cost(self, current_pos, move_vec):
        """计算生物力学成本 (用于训练Hand Agent)"""
        arm_vec = current_pos - self.fixed_point
        arm_len = np.linalg.norm(arm_vec)
        if arm_len < 1e-4: return 0
        arm_dir = arm_vec / arm_len
        v_rad = np.dot(move_vec, arm_dir) * arm_dir # 径向速度(伸缩)
        v_tan = move_vec - v_rad # 切向速度(扫动)
        s_tan = np.linalg.norm(v_tan)
        s_rad = np.linalg.norm(v_rad)
        # 惩罚远端的大幅度扫动 (Prevent full-screen sweep)
        p_sweep = self.w_sweep * (s_tan**2) * (1 + 2.0 * (arm_len / self.max_length_arm)**2)
        p_effort = self.w_effort * (s_rad**2)
        return p_sweep + p_effort

    def _get_obs(self):
        dist_bounds = np.array([
            self.robot_position[0],
            self.env_width - self.robot_position[0],
            self.robot_position[1],
            self.env_height - self.robot_position[1]
        ])
        
        flat_history = np.array(self.hand_history_buffer).flatten()
        if len(flat_history) < self.history_length * 2:
            flat_history = np.zeros(self.history_length * 2)

        obs = np.concatenate((
            self.robot_position,    # [0:2]
            self.hand_position,     # [2:4]
            flat_history,           # [4:24]
            [self.current_distance],# [24]
            dist_bounds,            # [25:29]
            [self.stride_robot],    # [29]
            self.fixed_point,       # [30:32]
            self.blocking_point     # [32:34]
        )).astype(np.float32)
        return obs

    def _get_info(self):
        return {
            "dist": self.current_distance,
            "steps": self.steps,
            "hand_pos": self.hand_position,
            "robot_pos": self.robot_position,
            "fixed_point": self.fixed_point,
            "blocking_point": self.blocking_point,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.stride_robot = np.random.uniform(*self.stride_robot_random)
        self.stride_hand = np.random.uniform(*self.stride_hand_random)
        
        self.robot_position = np.random.uniform(self.margin, [self.env_width-self.margin, self.env_height-self.margin])
        self.hand_position = np.random.uniform(self.margin, [self.env_width-self.margin, self.env_height-self.margin])
        self.fixed_point = np.array([self.env_width/2, self.env_height])
        
        self.hand_history_buffer.clear()
        for _ in range(self.history_length):
            self.hand_history_buffer.append(np.zeros(2))
            
        self.trajectory_points = [self.robot_position.copy()]
        self.current_distance = np.linalg.norm(self.robot_position - self.hand_position)
        self.pre_distance = self.current_distance
        self.steps = 0
        
        self._calculate_fix_point()
        vec = self.hand_position - self.fixed_point
        self.blocking_point = self.hand_position # 默认blocking point等于手
        
        return self._get_obs(), self._get_info()

    def _get_scripted_hand_move(self):
        """后备的脚本控制"""
        if random.random() < self.hand_move_epsilon:
            move = np.random.uniform(-1, 1, size=2)
            move = self.safe_normalize(move) * self.stride_hand
        else:
            vec = self.robot_position - self.hand_position
            move = self.safe_normalize(vec) * self.stride_hand
        return move

    def step(self, action):
        # 1. 预处理
        robot_move = np.zeros(2)
        hand_move_raw = np.zeros(2)
        bio_cost = 0.0
        
        if self.random_noise:
            action += np.random.normal(0, self.noise_sigma, size=action.shape)

        # 2. 决策逻辑 (关键修改部分)
        if self.training_mode == 'robot':
            # --- 训练 Robot: Robot 使用 Action, Hand 使用 Model 或 Script ---
            robot_move = action * self.stride_robot
            
            # [修改点] 检查是否有预训练的手掌模型
            if self.hand_model is not None:
                # 获取观测 (Hand Agent 也是用同样的 Env Observation)
                obs_for_hand = self._get_obs()
                # 预测动作
                hand_action, _ = self.hand_model.predict(obs_for_hand, deterministic=False) # False 保持一定的随机探索性
                hand_move_raw = hand_action * self.stride_hand
            else:
                # 如果没有模型，回退到脚本
                hand_move_raw = self._get_scripted_hand_move()
                
        else:
            # --- 训练 Hand: Hand 使用 Action, Robot 使用 Model ---
            hand_move_raw = action * self.stride_hand
            
            if self.robot_model is not None:
                obs_for_robot = self._get_obs() 
                robot_action, _ = self.robot_model.predict(obs_for_robot, deterministic=True)
                robot_move = robot_action * self.stride_robot
            else:
                robot_move = np.zeros(2) 

        # 3. 动作后处理 (Apply Filter)
        # 无论手是脚本控制还是模型控制，都要经过生物力学滤波器（模拟病理身体）
        if self.training_mode == 'hand':
            bio_cost = self._calculate_biomechanical_cost(self.hand_position, hand_move_raw)
            
        hand_move_final = self.bio_filter.apply(hand_move_raw, self.hand_position)

        # 4. 物理更新
        self.hand_position += hand_move_final
        self.hand_position = np.clip(self.hand_position, self.margin, [self.env_width-self.margin, self.env_height-self.margin])
        self.hand_history_buffer.append(hand_move_final)
        
        old_robot_pos = self.robot_position.copy()
        self.robot_position += robot_move
        self.trajectory_points.append(self.robot_position.copy())
        
        self._calculate_fix_point()
        vec_arm = self.fixed_point - self.hand_position
        dist_arm = np.linalg.norm(vec_arm)
        to_shoulder = self.safe_normalize(vec_arm)
        self.blocking_point = self.hand_position + to_shoulder * min(self.arm_blocking_length, dist_arm)

        self.steps += 1
        
        # 5. 奖励计算
        reward = 0
        terminated = False
        truncated = False
        done_reason = None
        
        self.current_distance = np.linalg.norm(self.robot_position - self.hand_position)
        
        # Robot Out of Bounds
        if np.any(self.robot_position <= self.margin) or \
           self.robot_position[0] >= self.env_width - self.margin or \
           self.robot_position[1] >= self.env_height - self.margin:
            if self.training_mode == 'robot': reward += self.reward_bound
            terminated = True
            done_reason = "Robot Out"

        # Caught
        if self.current_distance < self.distance_threshold_collision:
            if self.training_mode == 'robot': 
                reward += self.reward_robot_caught
            else: 
                reward += self.reward_hand_catch
            terminated = True
            done_reason = "Robot Caught"
            
        # Hit Arm
        if self._check_line_intersection(old_robot_pos, self.robot_position, self.hand_position, self.blocking_point):
            if self.training_mode == 'robot':
                reward += self.reward_arm_hit
            terminated = True
            done_reason = "Hit Arm"

        # Shaping
        dist_improvement = self.pre_distance - self.current_distance
        if self.training_mode == 'robot':
            reward += self.reward_step
        else:
            reward += dist_improvement * 5.0
            reward += self.reward_step
            reward -= bio_cost

        self.pre_distance = self.current_distance

        if self.steps >= self.max_steps:
            truncated = True
            if self.training_mode == 'robot': reward += self.reward_survival

        obs = self._get_obs()
        if self.random_noise:
            obs += np.random.normal(0, self.noise_sigma, size=obs.shape)
            
        info = self._get_info()
        info['bio_cost'] = bio_cost
        info['done_reason'] = done_reason
        
        return obs, reward, terminated, truncated, info

    # --- Render (保持不变或微调) ---
    def render(self, mode="human"):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                (int(self.env_width * self.cell_size), int(self.env_height * self.cell_size))
            )
            pygame.display.set_caption(f"Mode: {self.training_mode.upper()} | Hand: {self.hand_type}")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        canvas = pygame.Surface((self.window.get_width(), self.window.get_height()))
        canvas.fill(COLORS['bg'])
        
        to_px = lambda p: (int(p[0]*self.cell_size), int(p[1]*self.cell_size))

        hand_px = to_px(self.hand_position)
        robot_px = to_px(self.robot_position)
        fix_px = to_px(self.fixed_point)
        block_px = to_px(self.blocking_point)

        # Draw trajectory
        if len(self.trajectory_points) > 1:
            pts = [to_px(p) for p in self.trajectory_points[-50:]]
            if len(pts) > 1:
                pygame.draw.lines(canvas, COLORS['trajectory'], False, pts, 2)

        # Draw Arm
        pygame.draw.line(canvas, COLORS['arm_safe'], fix_px, block_px, 5)
        pygame.draw.line(canvas, COLORS['arm_block'], block_px, hand_px, 15)
        
        # Draw Entities
        pygame.draw.circle(canvas, (50,50,50), fix_px, 8) # Shoulder
        pygame.draw.circle(canvas, COLORS['hand'], hand_px, int(self.cell_size*0.3)) # Hand
        pygame.draw.circle(canvas, COLORS['robot'], robot_px, int(self.cell_size*0.25)) # Robot

        self.window.blit(canvas, (0,0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()




env = RehabilitationEnv()
env.random = False

w_env, h_env = 15, 10  # 环境的宽度和高度
# C:\Users\admin\Desktop\huifeng\RL\rlproject\src\model\model_500step.zip
# C:\Users\admin\Desktop\huifeng\RL\src\logs\best_model_sac88\best_model.zip
model = SAC.load(
    r"C:\Users\admin\Desktop\huifeng\RL\src\logs1\best_model_sac21\best_model.zip",
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
cx,cy = 0,0

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

        print(cx,cy)

        if now - last_trigger_time > 1/freq:

            # 获取机器人当前位姿（解包）
            *position_robot_world,z,rx,ry,rz = robot_control.get_robot_pose()

            position_robot_pixel = cali.world_to_pixel(position_robot_world)

            # position_robot_pixel = cx,cy
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
            
            robot = np.array([position_robot_env],dtype=np.float32).flatten()
            print(robot.shape)
            hand = np.array([position_hand_env],dtype=np.float32)
            hand_move = hand - last_hand
            stride_hand = np.linalg.norm(hand_move)
            print("stride_hand:",stride_hand)
            distance_to_object = np.linalg.norm(robot - hand)
            distance = np.array([distance_to_object],dtype=np.float32)
            boundary = np.array([
                robot[0],
                w_env-robot[0],
                robot[1],
                h_env-robot[1]
            ])
            last_action = np.array([last_action],dtype=np.float32)
            vec_arm = fixed_point - hand
            dist_arm = np.linalg.norm(vec_arm)
            to_shoulder = env.safe_normalize(vec_arm)
            blocking_point = hand + to_shoulder * min(5, dist_arm)

            flat_history = np.array(env.hand_history_buffer).flatten()
            if len(flat_history) < env.history_length * 2:
                flat_history = np.zeros(env.history_length * 2)

            stride_robot = env.stride_robot
            obs = np.concatenate((robot.flatten(),hand.flatten(),flat_history,distance.flatten(),boundary.flatten(),np.array([stride_robot]),np.array(fixed_point,dtype=np.float32),blocking_point.flatten()))

            action, _states = model.predict(obs, deterministic=True)
            env.hand_history_buffer.append(hand_move)
            last_action = action    
            print(f"obs:{obs}\n action:{action}")
            action_pixel = action * np.array([w/w_env,h/h_env])*stride_robot


            # cv2.circle(undistorted_frame, (int(position_robot_pixel[0]), int(position_robot_pixel[1])), 10, (0, 255, 0), -1)
            # cv2.circle(undistorted_frame, (int(position_hand_pixel[0]), int(position_hand_pixel[1])), 10, (255, 0, 0), -1)

            rx,ry,rz = 0.085,-0.027,4.637

            position_robot_pixel += np.array([action_pixel[0],action_pixel[1]])

            position_robot_pixel = np.clip(position_robot_pixel, 0, [w,h]).astype(int)
            position_robot_world = cali.pixel_to_world(position_robot_pixel)
            robot_control.move_robot([position_robot_world[0],position_robot_world[1],0.125,rx,ry,rz],1/freq)

            step += 1

            last_hand = hand
            last_trigger_time = now
            trajectory.append(position_robot_env)

            # --- 更新 env stride（你原来的逻辑） ---
            # 注意 obs[6] 可能是 numpy array 或标量，强制转 float
            # 但要防止除 0
            dist = float(obs[20]) if np.ndim(obs[20]) == 0 or np.ndim(obs[20])==1 else float(np.array(obs[20]).flatten()[0])

            # 避免除零
            dist = max(dist, 1e-6)

            # 指数调整
            base_stride = 2.5
            coef = np.exp(5-dist)  # 观察值越大，coef 越小
            # env.stride_robot = base_stride * coef

            # 可加上最大值限制
            # env.stride_robot = min(base_stride, env.stride_robot)


            # render 你的环境画面（保持）
            render.render(obs[:2], obs[2:4], fixed_point, trajectory,blocking_point.flatten())

            # ---------- 更新直方图：传入一个标量值 ----------
            # 把 dist 添加到 deque 并更新直方图
            if float(dist) < 2:
                terminated = True
            distance_list.append(float(dist))
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
