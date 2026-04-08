import gymnasium as gym
from gymnasium.spaces import Box,Dict
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import json
import time
from collections import deque
from APF import Advanced_APF


# --- Main Environment Class ---
class RehabilitationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, 
                 training_mode='robot', # 'robot' or 'hand'
                 robot_model=None,      # 训练 Hand 时需要的 Robot 对手
                 hand_model_paths=None,       # [新增] 训练 Robot 时需要的 Hand 对手
                 hand_type='healthy',   # 'healthy', 'parkinson', 'stroke'
                 render_mode=None):
        
        super().__init__()
        self.training_mode = training_mode
        self.robot_model = robot_model
        # self.hand_model_record = hand_model    # [新增] 保存手掌模型
        self.hand_type = hand_type
        self.render_mode = render_mode
        self.patient = None
        self.patient_locked = False

        self.apf_controller = Advanced_APF()
        self.residual_scale = 1
      # ========================================================
        # [核心修改 1]：构建 Hand 对手池 (Opponent Pool)
        # 永远把 None (代表基于规则的 Script Hand) 放在池子的第一个位置
        self.hand_model_pool = [None] 
        
        # 如果传入了历史模型的路径列表，让环境自己去逐个加载
        if hand_model_paths is not None:
            from stable_baselines3 import SAC,PPO # 确保环境内部能拿到 SAC
            for path in hand_model_paths:
                try:
                    # custom_objects 用于避免设备错乱 (Device mismatch)
                    model = PPO.load(path, custom_objects={'learning_rate': 0.0, 'optimizer_class': None})
                    self.hand_model_pool.append(model)
                except Exception as e:
                    print(f"无法加载历史对手模型 {path}, Error: {e}")
        
        # 当前局使用的 hand_model，默认先给个 None
        self.hand_model = None



        
        # --- Dimensions ---
        self.grid_size = 10
        self.cell_size = 50 
        self.env_width = self.grid_size * 1.5
        self.env_height = self.grid_size
        self.margin = 0.3
        
        # --- Physics & Arm ---
        self.max_length_arm = self.grid_size * 0.9
        self.fixed_point = np.array([self.env_width / 2, self.env_height])
        self.arm_blocking_length = 5
        
        # --- Filters ---
        # self.bio_filter = BiomechanicalFilter(self.hand_type)
        
        # --- Thresholds ---
        self.distance_threshold_collision = 2.5
        self.distance_threshold_penalty = 3.0
        
        # --- Rewards Config ---
        self.reward_hand_catch = 20
        self.reward_robot_caught = -20
        self.reward_arm_hit = -20
        self.reward_bound = -40
        self.reward_step = -0.2 if training_mode == 'hand' else 0.2
        self.reward_survival = 10
        
        # Biomechanical Cost Weights
        self.w_sweep = 10.0
        self.w_effort = 2.0
        
        # --- Movement Params ---
        self.stride_robot_random = [0.2, 1]
        self.stride_hand_random = [0.1, 1]
        self.hand_move_epsilon = 0.2
        
        self.max_steps = 100
        
        self.steps = 0
        
        # --- Spaces ---
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observation space definition
        self.history_length = 16
        # obs dim calculation: 
        # robot(2) + hand(2) + history(16) + dist(1) + bounds(4) + stride(1) + fix(2) + block(2) = 32
        self.obs_dim = 16+self.history_length*2
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # --- Internals ---
        self.robot_position = np.zeros(2)
        self.hand_position = np.zeros(2)
        self.blocking_point = np.zeros(2)
        self.hand_history_buffer = deque(maxlen=self.history_length)
        self.robot_history_buffer = deque(maxlen=self.history_length)
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

    # def _get_obs(self):
    #     dist_bounds = np.array([
    #         self.robot_position[0],
    #         self.env_width - self.robot_position[0],
    #         self.robot_position[1],
    #         self.env_height - self.robot_position[1]
    #     ])
        
    #     flat_history = np.array(self.hand_history_buffer).flatten()
    #     if len(flat_history) < self.history_length * 2:
    #         flat_history = np.zeros(self.history_length * 2)


    #     apf_force = self.apf_controller.compute_force(
    #         self.robot_position, 
    #         self.hand_position, 
    #         self.hand_dir
    #     )

    #     obs = np.concatenate((
    #         self.robot_position,    
    #         self.hand_position,     
    #         [self.current_distance],
    #         dist_bounds,            
    #         [self.stride_robot],    
    #         self.fixed_point,       
    #         self.blocking_point ,
    #         apf_force,
    #         flat_history,           

    #     )).astype(np.float32)
    #     return obs

    def _get_robot_obs(self):
        """Robot 的视角：关注 Hand 的历史"""
        dist_bounds = np.array([
            self.robot_position[0],
            self.env_width - self.robot_position[0],
            self.robot_position[1],
            self.env_height - self.robot_position[1]
        ])
        
        flat_hand_history = np.array(self.hand_history_buffer).flatten()
        if len(flat_hand_history) < self.history_length * 2:
            flat_hand_history = np.zeros(self.history_length * 2)

        # 计算 APF (Robot 需要知道物理引导)
        # apf_force = self.apf_controller.compute_force(
        #     self.robot_position, self.hand_position, self.hand_dir
        # )
        apf_force = np.zeros(2)

        obs = np.concatenate((
            self.robot_position,    
            self.hand_position,     
            [self.current_distance],
            dist_bounds,            
            [self.stride_robot],    
            self.fixed_point,       
            self.blocking_point,
            apf_force,              
            flat_hand_history,      # Robot 看 Hand 的历史
        )).astype(np.float32)
        return obs

    def _get_hand_obs(self):
        """Hand 的视角：关注 Robot 的历史"""
        # Hand 不需要知道 APF 力，也不需要知道 Robot 的边界距离(或者需要知道相对距离)
        # 这里为了简单，保持结构类似，但把 History 换成 Robot 的
        
        # 边界 (Hand 的边界)
        dist_bounds = np.array([
            self.hand_position[0],
            self.env_width - self.hand_position[0],
            self.hand_position[1],
            self.env_height - self.hand_position[1]
        ])
        
        flat_robot_history = np.array(self.robot_history_buffer).flatten()
        if len(flat_robot_history) < self.history_length * 2:
            flat_robot_history = np.zeros(self.history_length * 2)

        # Hand 不需要 APF Force，这里填 0 或者用 Robot 的速度代替
        # 为了保持输入维度一致(32维)，我们可以填入 relative_velocity
        # rel_vel = (self.robot_position - self.hand_position) # 简单替代
        rel_vel = np.zeros(2) # 简单替代

        obs = np.concatenate((
            self.hand_position,     # 自己的位置放在前 (Egocentric)
            self.robot_position,    # 对手的位置
            [self.current_distance],
            dist_bounds,            
            [self.stride_hand],     # 自己的步长
            self.fixed_point,       
            self.blocking_point,
            rel_vel,                # 替代 APF 的位置 (2维)
            flat_robot_history,     # Hand 看 Robot 的历史
        )).astype(np.float32)
        return obs

    # --- 4. 修改 _get_obs 统一接口 ---
    def _get_obs(self):
        """根据当前训练模式返回对应的主 Agent Obs"""
        if self.training_mode == 'robot':
            return self._get_robot_obs()
        else:
            return self._get_hand_obs()

    def _get_info(self):
        return {
            "dist": self.current_distance,
            "steps": self.steps,
            "hand_pos": self.hand_position,
            "robot_pos": self.robot_position,
            "fixed_point": self.fixed_point,
            "blocking_point": self.blocking_point,
        }



    def reset_patient(self):
        self.current_patient_param = self.patient.reset_randomly()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # if not self.patient_locked:
        #     self.reset_patient()

        
        # self.bio_filter.mode = random.choice(['healthy', 'parkinson']) 
        self.hand_model =  random.choice(self.hand_model_pool) 
        # self.hand_model =  self.hand_model_record

        self.stride_robot = np.random.uniform(*self.stride_robot_random)
        self.stride_hand = np.random.uniform(*self.stride_hand_random)
        self.arm_blocking_length = np.random.uniform(0, 1)
        # self.arm_blocking_length = 3



        # self.current_patient_param = self.patient.reset_randomly()


        
        self.robot_position = np.random.uniform(self.margin, [self.env_width-self.margin, self.env_height-self.margin])
        self.hand_position = np.random.uniform(self.margin, [self.env_width-self.margin, self.env_height-self.margin])

        self.fixed_point = np.array([self.env_width*random.gauss(0.5,0.15), self.env_height])
        
        self.hand_history_buffer.clear()
        for _ in range(self.history_length):
            self.hand_history_buffer.append(np.zeros(2))
        
        self.robot_history_buffer.clear()
        for _ in range(self.history_length):
            self.robot_history_buffer.append(np.zeros(2))
            
        self.trajectory_points = [self.robot_position.copy()]
        self.current_distance = np.linalg.norm(self.robot_position - self.hand_position)
        self.pre_distance = self.current_distance
        self.steps = 0
        
        self._calculate_fix_point()
        # 归一化
        self.hand_dir = (self.hand_position - self.fixed_point) / np.linalg.norm(self.hand_position - self.fixed_point)
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
    
    # 保持方向限幅函数
    def _clip(self, vec,max=1):
        if np.linalg.norm(vec) > max:
            return vec * max / np.linalg.norm(vec)
        else:
            return vec

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

            action_rl = action 
            
            # 2. APF 输出的是基准 (Base)
            action_base = self.apf_controller.compute_force(
                self.robot_position, 
                self.hand_position, 
                self.hand_dir
            )

            action_total = action_base + self.residual_scale * action_rl

            action_total = self._clip(action_total)

            # robot_move = action_total * self.stride_robot

            robot_move = action_rl * self.stride_robot
            
            # [修改点] 检查是否有预训练的手掌模型
            if self.hand_model is not None:
                # 获取观测 (Hand Agent 也是用同样的 Env Observation)
                obs_for_hand = self._get_hand_obs()
                # 预测动作
                hand_action, _ = self.hand_model.predict(obs_for_hand, deterministic=True) 
                hand_move_raw = hand_action * self.stride_hand
            else:
                # 如果没有模型，回退到脚本
                hand_move_raw = self._get_scripted_hand_move()
                # hand_move_raw = self.patient.get_velocity(self.hand_position,self.robot_position)* self.stride_hand
                
        else:
            # --- 训练 Hand: Hand 使用 Action, Robot 使用 Model ---
            hand_move_raw = action * self.stride_hand
            
            if self.robot_model is not None:
                obs_for_robot = self._get_robot_obs() 
                robot_action, _ = self.robot_model.predict(obs_for_robot, deterministic=True)
                robot_move = robot_action * self.stride_robot
            else:
                robot_move = np.zeros(2) 

        # 3. 动作后处理 (Apply Filter)
        # 无论手是脚本控制还是模型控制，都要经过生物力学滤波器（模拟病理身体）
        if self.training_mode == 'hand':
            bio_cost = self._calculate_biomechanical_cost(self.hand_position, hand_move_raw)
            
        # hand_move_final = self.bio_filter.apply(hand_move_raw, self.hand_position)
        hand_move_final = hand_move_raw


        # 4. 物理更新

        
        old_robot_pos = self.robot_position.copy()
        self.robot_position += robot_move
        self.trajectory_points.append(self.robot_position.copy())

        self.hand_position += hand_move_final
        self.hand_position = np.clip(self.hand_position, self.margin, [self.env_width-self.margin, self.env_height-self.margin])
        self.hand_history_buffer.append(hand_move_final)

        robot_actual_move = self.robot_position - old_robot_pos
        self.robot_history_buffer.append(robot_actual_move)
        
        self._calculate_fix_point()
        vec_arm = self.fixed_point - self.hand_position
        dist_arm = np.linalg.norm(vec_arm)
        to_shoulder = self.safe_normalize(vec_arm)
        self.blocking_point = self.hand_position + to_shoulder * min(self.arm_blocking_length, dist_arm)

        self.hand_dir  = -to_shoulder

        self.steps += 1
        
        # 5. 奖励计算
        reward = 0
        terminated = False
        truncated = False
        done_reason = None

        self.current_distance = np.linalg.norm(self.robot_position - self.hand_position)


        # 新增 Rehabilitation reward
        
        if self.training_mode == 'robot':
            z_min = 4
            z_max = 6
            # scenario 1: 患者的抓取速度很慢，robot要和hand的距离保持在3米以内
            if self.current_distance < z_min:
                rehab_reward = -0.1*np.exp(1.5*(z_min-self.current_distance)*2)

            elif self.current_distance < z_max:
                rehab_reward = 0.5

            else:
                rehab_reward= -0.3*(self.current_distance-z_max)


            rehab_reward = np.clip(rehab_reward,-10,10)
            reward += rehab_reward





        
        
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
            # reward -= dist_improvement * 2
        else:
            reward += dist_improvement * 0.5
            reward += self.reward_step
            # reward -= bio_cost

        # Penalize action
        reward -= 0.2 * np.linalg.norm(action)

        self.pre_distance = self.current_distance

        if self.steps >= self.max_steps:
            truncated = True
            if self.training_mode == 'robot': 
                reward += self.reward_survival
            else:
                reward -= self.reward_survival

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