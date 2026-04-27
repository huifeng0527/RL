import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import math
import pygame
import random
from collections import deque

from .renderer import render_aesthetic


class BiomechanicalFilter:
    """
    极简版生物力学滤波器 (Sim2Real 专用)
    仅保留核心病理特征：视运动延迟 (Latency) + 轻微震颤 (Micro-Tremor)
    物理基准：8 FPS (dt=0.125s), 1 env_unit = 5 cm
    """

    def __init__(self, mode='healthy'):
        self.mode = mode
        self.step_count = 0
        self.dt = 0.125  # 8 FPS 下的物理时间步长 (秒)

        # ==========================================
        # 1. 延迟参数 (模拟认知与神经传导迟缓)
        # ==========================================
        # 3帧延迟 = 3 * 125ms = 375ms (符合中风患者真实的反应滞后时间)
        self.delay_frames = 3
        self.delay_buffer = deque([np.zeros(2) for _ in range(self.delay_frames)], maxlen=self.delay_frames)

        # ==========================================
        # 2. 震颤参数 (模拟帕金森/原发性震颤)
        # ==========================================
        # 频率: 设定为 3.0 Hz (防 8FPS 采样混叠的理论安全上限)
        self.tremor_freq = 3.0 
        
        # 振幅: 0.15 units * 5 cm/unit = 0.75 cm (临床真实的轻微震颤幅度)
        self.tremor_amp = 0.05  

    def reset(self):
        """回合开始时清空时间与延迟队列"""
        self.delay_frames = random.randint(0, 3)
        self.delay_buffer = deque([np.zeros(2) for _ in range(self.delay_frames)], maxlen=self.delay_frames)
        self.step_count = 0




    def apply(self, ideal_action):
        """
        传入理想动作 (dx, dy)，返回叠加病理后的真实执行动作
        """
        self.step_count += 1
        action = np.array(ideal_action, dtype=float)

        # 健康模式：瞬间响应，完全贴合
        if self.mode == 'healthy':
            return action

        # ------------------------------------------------
        # 核心逻辑 1：应用神经传导延迟
        # ------------------------------------------------
        self.delay_buffer.append(action)
        delayed_action = self.delay_buffer.popleft()

        # ------------------------------------------------
        # 核心逻辑 2：应用周期性震颤与肌肉电噪声
        # ------------------------------------------------
        if self.mode in ['parkinson', 'impaired']:
            t_sec = self.step_count * self.dt
            
            # X和Y使用相差 pi/2 的正弦波，在物理空间上生成一个椭圆形的颤动轨迹
            tremor_x = self.tremor_amp * math.sin(2 * math.pi * self.tremor_freq * t_sec)
            tremor_y = self.tremor_amp * math.cos(2 * math.pi * self.tremor_freq * t_sec)
            
            # 叠加极微小的独立高斯白噪声，模拟肌肉肌电噪声 (0.02 units = 1mm)
            noise = np.random.normal(0, 0.02, 2)
            
            final_action = delayed_action + np.array([tremor_x, tremor_y]) + noise
            return final_action

        # 如果 mode 是 'stroke' 或其他，只返回延迟
        return delayed_action


OBS_SCALAR_DIM = 10
HISTORY_LENGTH = 16
HISTORY_CHANNELS = 2

COLORS = {
    "bg": (245, 247, 250),
    "trajectory": (67, 97, 238),
    "arm_safe": (44, 160, 44),
    "arm_block": (230, 57, 70),
    "hand": (46, 196, 182),
    "robot": (255, 159, 28),
}


# --- Main Environment Class ---
class RehabilitationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self,
                 training_mode='robot', # 'robot' or 'hand'
                 robot_model=None,      # 训练 Hand 时需要的 Robot 对手
                 hand_model=None,       # 直接传入的 hand 模型（优先级高于 pool）
                 hand_model_paths=None, # 训练 Robot 时需要的 Hand 对手路径列表
                 pathology_mode='healthy', # 'healthy', 'parkinson', 'stroke', 'ataxia'
                 render_mode=None):

        super().__init__()
        self.training_mode = training_mode
        self.robot_model = robot_model
        self.render_mode = render_mode
        self.pathology_mode = pathology_mode

        # Biomechanical filter for simulating pathology
        self.biomech_filter = BiomechanicalFilter(mode=pathology_mode)

        # ========================================================
        # Hand 模型加载逻辑：
        # - hand_model 参数优先级最高：直接使用这一个模型
        # - 否则用 hand_model_paths 构建对手池（None=脚本手）
        # ========================================================
        self.hand_model = hand_model

        if hand_model is not None:
            # 优先级1：直接传入的 hand_model（独占模式）
            self.hand_model_pool = [hand_model]
        else:
            # 优先级2：使用对手池（仅在训练 Robot 时需要）
            self.hand_model_pool = []
            if hand_model_paths is not None:
                from stable_baselines3 import SAC, PPO
                for path in hand_model_paths:
                    try:
                        model = PPO.load(path, custom_objects={'learning_rate': 0.0, 'optimizer_class': None}, verbose=0)
                        self.hand_model_pool.append(model)
                    except Exception as e:
                        print(f"无法加载历史对手模型 {path}, Error: {e}")



        
        # --- Dimensions ---
        self.grid_size = 10
        self.cell_size = 50 
        self.env_width = self.grid_size * 1.5
        self.env_height = self.grid_size
        self.margin = 0.3
        
        # --- Physics & Arm ---
        self.max_length_arm = self.grid_size * 0.9
        self.fixed_point = np.array([self.env_width / 2, self.env_height])

        # --- Thresholds ---
        self.distance_threshold_collision = 2
        # self.distance_threshold_penalty = 3.0\
        
        # --- Rewards Config ---
        self.reward_hand_catch = 30
        self.reward_robot_caught = -40
        self.reward_arm_hit = -20
        self.reward_bound = -50
        self.reward_step = -0.2 if training_mode == 'hand' else 0.2
        # self.reward_survival = 10
        
        # Biomechanical Cost Weights
        self.w_sweep = 10.0
        self.w_effort = 2.0
        
        # --- Movement Params ---
        self.stride_robot_random = [0.6, 0.8]
        # dual: [0.3, 0.4]

        self.stride_hand_random = [0.3, 0.4]
        self.hand_move_epsilon = 0.2
        
        self.max_steps = 100
        
        self.steps = 0
        
        # --- Spaces ---
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observation space definition
        self.history_length = HISTORY_LENGTH
        # obs dim calculation:
        # actor position(2) + opponent position(2) + dist(1) + bounds(4) + stride(1) + history(16 * 2)
        self.obs_dim = OBS_SCALAR_DIM + self.history_length * HISTORY_CHANNELS
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # --- Internals ---
        self.robot_position = np.zeros(2)
        self.hand_position = np.zeros(2)
        self.blocking_point = np.zeros(2)

        # ========================================================
        # [核心新增]：Hand 物理约束状态变量
        # ========================================================
        self.last_hand_actual_move = np.zeros(2, dtype=np.float32)

        # ========================================================
        # [核心新增]：Robot 动作平滑惩罚状态变量
        # ========================================================
        self.last_robot_action = np.zeros(2, dtype=np.float32)
        self._jerk_penalty = 0.0

        # 旁路标志：外部直接控制 hand_position 时跳过物理约束
        self._bypass_hand_physics = False

        self.hand_history_buffer = deque(maxlen=self.history_length)
        self.robot_history_buffer = deque(maxlen=self.history_length)
        self.trajectory_points = []
        
        self.window = None
        self.clock = None
        self.random_noise = True
        self.noise_sigma = 0.05

    def _seed_history_buffer(self, buffer):
        buffer.clear()
        for _ in range(self.history_length):
            buffer.append(np.zeros(2, dtype=np.float32))

    def _sample_episode_parameters(self):
        # ========================================================
        # PFSP: Priority Fictitious Self-Play (仅在训练 Robot 时使用)
        # 当训练 Hand 时，不需要采样 hand_model
        # ========================================================
        if self.training_mode == 'robot' and len(self.hand_model_pool) > 0:
            pool_size = len(self.hand_model_pool)

            if pool_size == 1:
                self.hand_model = self.hand_model_pool[0]
            else:
                p = np.zeros(pool_size)
                p[0] = 0.20   # 20% 打 Script Hand (保底基本功)
                p[-1] = 0.50  # 50% 打最新 Hand (突破上限)

                if pool_size > 2:
                    remaining_prob = (1 - p[0] - p[-1]) / (pool_size - 2)
                    p[1:-1] = remaining_prob  # 30% 平分给历史模型
                else:
                    p[-1] = 1-p[0]  # 只有 script + 1个模型时

                self.hand_model = np.random.choice(self.hand_model_pool, p=p)

        self.stride_robot = np.random.uniform(*self.stride_robot_random)
        self.stride_hand = np.random.uniform(*self.stride_hand_random)
        self.arm_blocking_length = np.random.uniform(0, 1)

    def _sample_initial_positions(self):
        bounds = [self.env_width - self.margin, self.env_height - self.margin]
        self.robot_position = np.random.uniform(self.margin, bounds)
        self.hand_position = np.random.uniform(self.margin, bounds)
        self.fixed_point = np.array([self.env_width * random.gauss(0.5, 0.15), self.env_height])

    def _apply_obs_noise(self, obs):
        if not self.random_noise:
            return obs
        return obs + np.random.normal(0, self.noise_sigma, size=obs.shape)

    def _get_rehab_reward(self):
        z_min = 4
        z_max = 6
        if self.current_distance < z_min:
            reward = -0.1 * np.exp(1.5 * (z_min - self.current_distance) * 2)
        elif self.current_distance < z_max:
            reward = 0.5
        else:
            reward = -0.3 * (self.current_distance - z_max)
        return float(np.clip(reward, -1, 1))

    def _robot_out_of_bounds(self):
        return bool(
            np.any(self.robot_position <= self.margin)
            or self.robot_position[0] >= self.env_width - self.margin
            or self.robot_position[1] >= self.env_height - self.margin
        )

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
            # self.fixed_point,       
            # self.blocking_point,
            # apf_force,              
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
            # self.fixed_point,       
            # self.blocking_point,
            # rel_vel,                # 替代 APF 的位置 (2维)
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
        self._sample_episode_parameters()
        self._sample_initial_positions()
        self._seed_history_buffer(self.hand_history_buffer)
        self._seed_history_buffer(self.robot_history_buffer)
        self.trajectory_points = [self.robot_position.copy()]
        self.current_distance = np.linalg.norm(self.robot_position - self.hand_position)
        self.pre_distance = self.current_distance
        self.steps = 0

        # [核心新增]：回合重置时清空物理惯性状态
        self.last_hand_actual_move = np.zeros(2, dtype=np.float32)
        self.last_robot_action = np.zeros(2, dtype=np.float32)
        self._jerk_penalty = 0.0
        self._bypass_hand_physics = False

        # Reset biomechanical filter for new episode
        self.biomech_filter.reset()

        self._calculate_fix_point()
        # 归一化
        self.hand_dir = (self.hand_position - self.fixed_point) / np.linalg.norm(self.hand_position - self.fixed_point)
        self.blocking_point = self.hand_position # 默认blocking point等于手
        
        return self._get_obs(), self._get_info()

    def _get_scripted_hand_move(self):
        """后备的脚本控制（使用更大的步长，比 RL Hand 更激进）"""
        # Scripted hand 用 0.5~0.7 的较大步长，更具威胁性
        scripted_stride = random.uniform(0.5, 0.7)
        if random.random() < self.hand_move_epsilon:
            move = np.random.uniform(-1, 1, size=2)
            move = self.safe_normalize(move) * scripted_stride
        else:
            vec = self.robot_position - self.hand_position
            move = self.safe_normalize(vec) * scripted_stride
        return move
    
    def _resolve_robot_move(self, action):
        if self.training_mode == 'hand':
            if self.robot_model is None:
                return np.zeros(2)
            obs_for_robot = self._get_robot_obs()
            robot_action, _ = self.robot_model.predict(obs_for_robot, deterministic=False)
            return robot_action * self.stride_robot

        # Robot mode: use RL action directly
        return action * self.stride_robot

    def _resolve_hand_move(self, action):
        # 旁路模式：外部直接控制 hand_position，跳过物理约束
        if getattr(self, '_bypass_hand_physics', False):
            return np.zeros(2)

        # 1. 获取手的"大脑意图" (Raw Action)
        if self.training_mode == 'hand':
            hand_intent = action * self.stride_hand
        elif self.hand_model is None:
            hand_intent = self._get_scripted_hand_move()
        else:
            obs_for_hand = self._get_hand_obs()
            hand_action, _ = self.hand_model.predict(obs_for_hand, deterministic=True)
            hand_intent = hand_action * self.stride_hand

        # ========================================================
        # 2. 物理约束 I：一阶惯性低通滤波 (Muscle Inertia)
        # ========================================================
        alpha = 0.3
        smoothed_move = alpha * hand_intent + (1.0 - alpha) * self.last_hand_actual_move

        # ========================================================
        # 3. 物理约束 II：最大加速度截断 (Acceleration Clipping)
        # ========================================================
        max_accel = 0.15
        delta_v = smoothed_move - self.last_hand_actual_move
        accel_magnitude = np.linalg.norm(delta_v)

        if accel_magnitude > max_accel:
            delta_v = (delta_v / accel_magnitude) * max_accel

        final_physics_move = self.last_hand_actual_move + delta_v

        # ========================================================
        # 4. 更新状态并返回最终的物理位移
        # ========================================================
        self.last_hand_actual_move = final_physics_move.copy()
        return final_physics_move

    def _compute_reward_and_done(self, old_robot_pos):
        reward = 0.0
        terminated = False
        truncated = False
        done_reason = None

        self.current_distance = np.linalg.norm(self.robot_position - self.hand_position)

        if self.training_mode == 'robot':
            reward += self._get_rehab_reward()

        if self._robot_out_of_bounds():
            if self.training_mode == 'robot':
                reward += self.reward_bound
            terminated = True
            done_reason = "Robot Out"

        if self.current_distance < self.distance_threshold_collision:
            reward += self.reward_robot_caught if self.training_mode == 'robot' else self.reward_hand_catch
            terminated = True
            done_reason = "Robot Caught"

        # if self._check_line_intersection(old_robot_pos, self.robot_position, self.hand_position, self.blocking_point):
        #     if self.training_mode == 'robot':
        #         reward += self.reward_arm_hit
        #     terminated = True
        #     done_reason = "Hit Arm"

        dist_improvement = self.pre_distance - self.current_distance
        if self.training_mode == 'robot':
            reward += self.reward_step
            # Robot 动作平滑惩罚：惩罚动作的突变（jerk）
            # 动作变化越大，惩罚越重（避免 Robot 学会"抽搐"策略）
            jerk_penalty = -0.1 * self._jerk_penalty
            reward += jerk_penalty
        else:
            # Hand 必须主动追逐，不能镜像对峙
            # 1. 接近奖励：越近越高（主动追击的内驱力）
            reward += dist_improvement * 0.5
            # 2. 距离亲近奖励：当前距离越近 bonus 越大（防止躺平对峙）
            # max_distance ≈ 对角线 ≈ 18, min_distance = 0
            proximity_reward = max(0, (18 - self.current_distance) / 18) * 0.1
            reward += proximity_reward
            # 3. 步进惩罚：促使 hand 尽快抓到你
            reward += self.reward_step

        self.pre_distance = self.current_distance

        if self.steps >= self.max_steps:
            truncated = True

        return reward, terminated, truncated, done_reason

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        bio_cost = 0.0

        if self.random_noise:
            action += np.random.normal(0, self.noise_sigma, size=action.shape)

        robot_move = self._resolve_robot_move(action)
        hand_move_raw = self._resolve_hand_move(action)

        # Apply biomechanical filter to simulate pathology
        hand_move_final = self.biomech_filter.apply(hand_move_raw)

        if self.training_mode == 'hand':
            bio_cost = self._calculate_biomechanical_cost(self.hand_position, hand_move_final)

        old_robot_pos = self.robot_position.copy()
        self.robot_position += robot_move
        self.trajectory_points.append(self.robot_position.copy())

        self.hand_position += hand_move_final
        self.hand_position = np.clip(self.hand_position, self.margin, [self.env_width-self.margin, self.env_height-self.margin])
        self.hand_history_buffer.append(hand_move_final)

        robot_actual_move = self.robot_position - old_robot_pos
        self.robot_history_buffer.append(robot_actual_move)

        # Robot 动作平滑惩罚：计算 jerk（加速度变化率）并存储
        self._jerk_penalty = np.linalg.norm(robot_actual_move - self.last_robot_action)
        # 更新状态
        self.last_robot_action = robot_actual_move.copy()
        
        self._calculate_fix_point()
        vec_arm = self.fixed_point - self.hand_position
        dist_arm = np.linalg.norm(vec_arm)
        to_shoulder = self.safe_normalize(vec_arm)
        self.blocking_point = self.hand_position + to_shoulder * min(self.arm_blocking_length, dist_arm)

        self.hand_dir  = -to_shoulder

        self.steps += 1

        reward, terminated, truncated, done_reason = self._compute_reward_and_done(old_robot_pos)

        obs = self._apply_obs_noise(self._get_obs())
        info = self._get_info()
        info['bio_cost'] = bio_cost
        info['done_reason'] = done_reason
        
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(
                (int(self.env_width * self.cell_size), int(self.env_height * self.cell_size))
            )
            pygame.display.set_caption(f"Mode: {self.training_mode.upper()} | RehabilitationEnv")
            self.clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        render_aesthetic(
            self.robot_position,
            self.hand_position,
            self.fixed_point,
            self.trajectory_points,
            grid_size=self.grid_size,
            cell_size=self.cell_size,
            window=self.window
        )

        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
