"""
统一的仿真环境实现
基于Gymnasium的2D机器人追逐手部环境
"""
import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import pygame
import random
import os
import json
import time

# Define colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)  # Color for the trajectory


class CustomEnv(gym.Env):
    """
    自定义环境：机器人追逐手部的2D仿真环境
    
    观察空间：
    - robot_position (2): 机器人位置 (x, y)
    - hand_position (2): 手部位置 (x, y)
    - last_action (2): 上一步动作
    - current_distance (1): 当前距离
    - boundary (1): 到边界的最近距离
    - dist_arm (1): 到手臂的距离
    - fixed_point (2): 固定点位置
    - stride_robot (1): 机器人步长
    - stride_hand (1): 手部步长
    - env_size (2): 环境尺寸
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, config=None):
        """
        初始化环境
        
        Args:
            render_mode: 渲染模式
            config: 配置字典，如果为None则使用默认值
        """
        super().__init__()
        
        # 从配置加载参数，如果没有配置则使用默认值
        if config is None:
            config = self._get_default_config()
        
        self.grid_size = config.get('grid_size', 10)
        self.env_width = self.grid_size * 1.5
        self.env_height = self.grid_size

        self.pause = False
        self.domain_randomization = config.get('domain_randomization', False)
        self.render_mode = render_mode

        # 距离阈值
        self.distance_threshold_penalty = config.get('distance_threshold_penalty', 5)
        self.distance_threshold_collision = config.get('distance_threshold_collision', 1.5)
        self.distance_threshold_arm = config.get('distance_threshold_arm', 3)
        
        # 奖励参数
        self.penalty_factor = config.get('penalty_factor', 5)
        self.distance_reward_factor = config.get('distance_reward_factor', 2)
        self.smooth_action_penalty = config.get('smooth_action_penalty', 2)
        self.margin = config.get('margin', 0.3)
        self.reward_arm = config.get('reward_arm', -100)
        self.reward_hand = config.get('reward_hand', -100)
        self.reward_bound = config.get('reward_bound', -200)
        self.reward_max_step = config.get('reward_max_step', 200)
        self.reward_step = config.get('reward_step', 10)
        
        # 步长范围
        self.stride_robot_random = config.get('stride_robot_range', [1, 3])
        self.stride_hand_random = config.get('stride_hand_range', [0.6, 1])
        self.hand_move_epsilon = config.get('hand_move_epsilon', 0.1)
        
        # 环境参数
        self.current_distance = 0
        self.max_steps = config.get('max_steps', 50)
        
        # Action space (dx, dy)
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Observation space
        self.observation_shape = 2+2+2+1+1+1+2+1+1+2  # 总共17维
        self.observation_space = Box(
            low=0, 
            high=np.array([
                self.env_width, self.env_height,  # robot position
                self.env_width, self.env_height,  # hand position
                1, 1,  # last action
                (2**0.5)*self.grid_size,  # current distance
                0.5*self.grid_size,  # boundary
                0.5*self.grid_size,  # dist_arm
                2*self.grid_size, self.grid_size,  # fixed_point
                self.grid_size,  # stride_robot
                self.stride_hand_random[1],  # stride_hand
                self.env_width, self.env_height  # env_size
            ]), 
            shape=(self.observation_shape,), 
            dtype=np.float32
        )

        self.random = config.get('random', True)
        
        # 渲染相关
        self.window = None
        self.clock = None
        self.cell_size = config.get('cell_size', 50)
        self.trajectory_points = []
        self.dist_arm = 0
        
        # 手部图像路径（如果存在）
        self.hand_image_path = config.get('hand_image_path', None)

    def _get_default_config(self):
        """返回默认配置"""
        return {
            'grid_size': 10,
            'domain_randomization': False,
            'distance_threshold_penalty': 5,
            'distance_threshold_collision': 1.5,
            'distance_threshold_arm': 3,
            'penalty_factor': 5,
            'distance_reward_factor': 2,
            'smooth_action_penalty': 2,
            'margin': 0.3,
            'reward_arm': -100,
            'reward_hand': -100,
            'reward_bound': -200,
            'reward_max_step': 200,
            'reward_step': 10,
            'stride_robot_range': [1, 3],
            'stride_hand_range': [0.6, 1],
            'hand_move_epsilon': 0.1,
            'max_steps': 50,
            'random': True,
            'cell_size': 50,
            'hand_image_path': None
        }

    def dist_point_to_segment_correct(self, P, A, B, eps=1e-12):
        """
        计算点到线段的距离
        
        Args:
            P: 点坐标
            A: 线段起点
            B: 线段终点
            eps: 小值阈值
            
        Returns:
            distance: 距离
            closest_point: 最近点
            t: 参数t
            case: 情况描述
        """
        P = np.asarray(P, dtype=float)
        A = np.asarray(A, dtype=float)
        B = np.asarray(B, dtype=float)
        v = B - A
        w = P - A
        vv = np.dot(v, v)
        if vv <= eps:
            # A and B coincide: treat as point A
            C = A.copy()
            d = np.linalg.norm(P - A)
            t = 0.0
            case = 'endpoint_A'
        else:
            t = np.dot(w, v) / vv
            if t < 0.0:
                C = A
                d = np.linalg.norm(P - A)
                case = 'before_A'
            elif t > 1.0:
                C = B
                d = np.linalg.norm(P - B)
                case = 'after_B'
            else:
                C = A + t * v
                d = np.linalg.norm(P - C)
                case = 'on_segment'
        return float(d), C, float(t), case

    def _get_obs(self):
        """获取观察值"""
        return np.concatenate((
            [self.robot_position], 
            [self.hand_position],
            [self.last_action],
            [np.array([self.current_distance])],
            [np.array([min(
                self.robot_position[0],
                self.robot_position[1],
                self.env_width - self.robot_position[0],
                self.env_height - self.robot_position[1]
            )])],
            [np.array([self.dist_arm])],
            [self.fixed_point],
            [np.array([self.stride_robot])],
            [np.array([self.stride_hand])],
            [np.array([self.env_width, self.env_height])]
        ))

    def _get_info(self):
        """获取信息字典"""
        return {
            "distance_to_hand": self.current_distance,
            "robot_position": self.robot_position,
            "hand_position": self.hand_position,
            'distance_arm': self.dist_arm,
            "fix_point": self.fixed_point,
        }

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        self.distance = []
        self.stride_robot = np.random.uniform(*self.stride_robot_random)
        self.stride_hand = np.random.uniform(*self.stride_hand_random)
        self.distance_threshold_collision = np.random.uniform(2, 3)
        self.distance_threshold_penalty = np.random.uniform(3, 4)
        
        self.noise_obs_sigma = np.random.uniform(0, 0.1)
        self.noise_action_sigma = np.random.uniform(0, 0.1)
        
        self.robot_position = np.random.uniform(
            self.margin, 
            [self.env_width - self.margin, self.env_height - self.margin]
        )
        self.hand_position = np.random.uniform(
            self.margin, 
            [self.env_width - self.margin, self.env_height - self.margin]
        )
        
        self.current_distance = np.linalg.norm(self.robot_position - self.hand_position)
        self.pre_distance = self.current_distance
        self.last_action = np.zeros(2)
        self.steps = 0
        self.trajectory_points = [self.robot_position.copy()]
        
        self.fixed_point = np.array([
            self.grid_size * random.uniform(0.2, 1.3),
            self.grid_size
        ])
        return self._get_obs(), self._get_info()

    def _reward(self, action):
        """
        计算奖励
        
        Returns:
            reward: 奖励值
            terminated: 是否终止
            truncated: 是否截断
            done_reason: 终止原因
        """
        terminated = False
        truncated = False
        reward = 0
        done_reason = None

        # 计算到手臂的距离
        self.dist_arm = self.dist_point_to_segment_correct(
            self.robot_position, 
            self.hand_position, 
            self.fixed_point
        )[0]
        
        if self.dist_arm < self.distance_threshold_arm:
            reward += self.reward_arm
            terminated = True
            done_reason = "arm too short"

        # 边界惩罚
        if (np.any(self.robot_position <= self.margin) or 
            (self.env_height - self.robot_position[1] <= self.margin) or 
            self.env_width - self.robot_position[0] <= self.margin):
            reward += self.reward_bound
            terminated = True
            done_reason = "out of bounds"

        # 距离奖励
        self.current_distance = np.linalg.norm(self.robot_position - self.hand_position)
        self.distance.append(self.current_distance)
        reward += (self.current_distance - self.pre_distance) * self.distance_reward_factor
        self.pre_distance = self.current_distance

        # 障碍物处理
        if self.current_distance < self.distance_threshold_collision:
            reward += self.reward_hand
            terminated = True
            done_reason = "collision with obstacle"
        elif self.current_distance < self.distance_threshold_penalty:
            reward -= self.penalty_factor * (self.distance_threshold_penalty - self.current_distance)

        # 动作平滑性惩罚
        reward -= self.smooth_action_penalty * np.linalg.norm(action - self.last_action)

        # 步数奖励
        reward += self.reward_step

        # 最大步数检查
        if self.steps >= self.max_steps:
            reward += self.reward_max_step
            truncated = True

        return reward, terminated, truncated, done_reason

    def _get_hand_movement(self):
        """获取手部移动"""
        if random.random() < self.hand_move_epsilon:
            move_hand = np.random.uniform(-1, 1, size=2)
        else:
            dir_vector = self.robot_position - self.hand_position
            if np.linalg.norm(dir_vector) > 0:
                dir_vector /= np.linalg.norm(dir_vector)
            move_hand = dir_vector * self.stride_hand
        return move_hand

    def step(self, action):
        """执行一步"""
        if self.random:
            action += np.random.normal(
                0, 
                self.noise_action_sigma, 
                size=self.action_space.shape
            )

        move_hand = self._get_hand_movement()
        self.hand_position += move_hand
        self.hand_position = np.clip(
            self.hand_position, 
            self.margin, 
            [self.env_width - self.margin, self.env_height - self.margin]
        )

        self.robot_position += action * self.stride_robot
        self.trajectory_points.append(self.robot_position.copy())
        self.steps += 1

        reward, terminated, truncated, done_reason = self._reward(action)
        info = self._get_info()
        info['done_reason'] = done_reason
        info['distance_mean'] = np.mean(self.distance) if self.distance else 0
        observation = self._get_obs()
        
        if self.random:
            observation += np.random.normal(
                0, 
                self.noise_obs_sigma, 
                size=self.observation_shape
            )

        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        """渲染环境"""
        pygame.display.init()
        self.window = pygame.display.set_mode(
            (int(self.grid_size * self.cell_size), int(self.grid_size * self.cell_size))
        )
        pygame.display.set_caption("CustomEnv")
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                import sys
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                self.hand_position = np.array([mouse_x/self.cell_size, mouse_y/self.cell_size])
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.pause = not self.pause

        canvas = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        canvas.fill(WHITE)
        
        # 加载手部图像（如果存在）
        if self.hand_image_path and os.path.exists(self.hand_image_path):
            virus_image = pygame.image.load(self.hand_image_path).convert_alpha()
        else:
            # 创建一个简单的占位图像
            virus_image = pygame.Surface((self.cell_size * 2, self.cell_size * 2))
            virus_image.fill(GREEN)
        
        robot_image = pygame.transform.scale(
            virus_image, 
            (int(self.cell_size * 2), int(self.cell_size * 2))
        )
        
        # 绘制轨迹
        if len(self.trajectory_points) > 1:
            scaled_points = []
            for point in self.trajectory_points:
                scaled_points.append((
                    int(point[0] * self.cell_size), 
                    int(point[1] * self.cell_size)
                ))
            pygame.draw.lines(canvas, BLUE, False, scaled_points, 2)
            for point_coord in scaled_points:
                pygame.draw.circle(canvas, BLUE, point_coord, 3)

        # 绘制机器人
        pygame.draw.circle(
            canvas,
            RED,
            (int(self.robot_position[0] * self.cell_size), 
             int(self.robot_position[1] * self.cell_size)),
            int(self.cell_size * 0.2)
        )
        
        # 绘制手部
        canvas.blit(
            robot_image, 
            (int((self.hand_position[0] - 1) * self.cell_size), 
             int((self.hand_position[1] - 1) * self.cell_size + 1))
        )
        pygame.draw.circle(
            canvas,
            GREEN, 
            (int(self.hand_position[0] * self.cell_size), 
             int((self.hand_position[1] * self.cell_size + 1))), 
            int(self.cell_size * 0.2)
        )

        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        time.sleep(0.5)

    def load_args(self, args):
        """加载参数（兼容性方法）"""
        pass

    def save_args(self, path):
        """保存环境参数到JSON文件"""
        env_args = {
            "grid_size": self.grid_size,
            "distance_threshold_penalty": self.distance_threshold_penalty,
            "distance_threshold_collision": self.distance_threshold_collision,
            "penalty_factor": self.penalty_factor,
            "distance_reward_factor": self.distance_reward_factor,
            "smooth_action_penalty": self.smooth_action_penalty,
            "max_steps": self.max_steps,
            "margin": self.margin,
            "reward_step": self.reward_step,
            "reward_max_step": self.reward_max_step,
            "reward_bound": self.reward_bound,
            "reward_arm": self.reward_arm,
            "reward_hand": self.reward_hand,
            "stride_robot_range": self.stride_robot_random,
            "stride_hand_range": self.stride_hand_random,
            "move_hand_epsilon": self.hand_move_epsilon,
        }
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "env_args.json"), "w") as f:
            json.dump(env_args, f, indent=4)

    def close(self):
        """关闭环境"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

