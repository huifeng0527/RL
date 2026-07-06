import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
import math
import os
import pygame
import random
from collections import deque

from .renderer import render_aesthetic
from .observation_schema import (
    DEFAULT_HISTORY_LENGTH,
    HISTORY_CHANNELS,
    INTERACTION_HISTORY_CHANNELS,
    OBS_SCALAR_DIM,
    adapt_history_obs,
    model_obs_dim,
    obs_dim,
)


class BiomechanicalFilter:
    """
     (Sim2Real )
    ?(Latency) +  (Micro-Tremor)
    ? FPS (dt=0.125s), 1 env_unit = 5 cm
    """

    def __init__(self, mode='healthy'):
        self.mode = mode
        self.step_count = 0
        self.dt = 0.125  # 8 FPS  (?

        # ==========================================
        # 1.  (?
        # ==========================================
        # 3?= 3 * 125ms = 375ms ()
        self.delay_frames = 3
        self.delay_buffer = deque([np.zeros(2) for _ in range(self.delay_frames)], maxlen=self.delay_frames)

        # ==========================================
        # 2.  (??
        # ==========================================
        # : ?3.0 Hz (?8FPS ?
        self.tremor_freq = 3.0 
        
        # : 0.15 units * 5 cm/unit = 0.75 cm (?
        self.tremor_amp = 0.05  

    def reset(self):
        """Reset biomechanical filter state."""
        self.delay_frames = random.randint(0, 3)
        self.delay_buffer = deque([np.zeros(2) for _ in range(self.delay_frames)], maxlen=self.delay_frames)
        self.step_count = 0




    def apply(self, ideal_action):
        """
         (dx, dy)?
        """
        self.step_count += 1
        action = np.array(ideal_action, dtype=float)

        # 
        if self.mode == 'healthy':
            return action

        # ------------------------------------------------
        #  1?
        # ------------------------------------------------
        if self.delay_frames <= 0:
            delayed_action = action
        else:
            self.delay_buffer.append(action)
            delayed_action = self.delay_buffer.popleft()

        # ------------------------------------------------
        #  2?
        # ------------------------------------------------
        if self.mode in ['parkinson', 'impaired']:
            t_sec = self.step_count * self.dt
            
            # # X pi/2 ?
            # tremor_x = self.tremor_amp * math.sin(2 * math.pi * self.tremor_freq * t_sec)
            # tremor_y = self.tremor_amp * math.cos(2 * math.pi * self.tremor_freq * t_sec)
            
            #  (0.02 units = 1mm)
            noise = np.random.normal(0, 0.02, 2)
            
            final_action = delayed_action + noise
            return final_action

        #  mode ?'stroke' ?
        return delayed_action


HISTORY_LENGTH = DEFAULT_HISTORY_LENGTH

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
                 robot_model=None,      #  Hand  Robot 
                 hand_model=None,       # ?hand  pool?
                 hand_model_paths=None, #  Robot  Hand 
                 pathology_mode='healthy', # 'healthy', 'parkinson', 'stroke', 'ataxia'
                 render_mode=None,
                 include_opponent_id=False,
                 robot_opponent_id_dim=0,
                 history_length=HISTORY_LENGTH,
                 history_mode="motion"):

        super().__init__()
        self.training_mode = training_mode
        self.robot_model = robot_model
        self.render_mode = render_mode
        self.pathology_mode = pathology_mode
        self.include_opponent_id = include_opponent_id
        self.robot_opponent_id_dim = int(robot_opponent_id_dim)
        self.history_length = int(history_length)
        self.history_mode = str(history_mode)
        if self.history_mode not in {"motion", "interaction"}:
            raise ValueError(f"Unsupported history_mode={history_mode!r}")
        self.history_channels = INTERACTION_HISTORY_CHANNELS if self.history_mode == "interaction" else HISTORY_CHANNELS
        self.current_opponent_id = 0
        self.current_opponent_one_hot_dim = 1

        # Biomechanical filter for simulating pathology
        self.biomech_filter = BiomechanicalFilter(mode=pathology_mode)

        # ========================================================
        # Hand ?
        # - hand_model ?
        # - ?hand_model_paths None=
        # ========================================================
        self.hand_model = hand_model
        self.hand_model_paths = list(hand_model_paths or [])
        self.hand_model_names = []

        if hand_model is not None:
            # ? hand_model
            self.hand_model_pool = [hand_model]
            self.hand_model_names = ["direct_hand"]
        else:
            # ??Robot 
            self.hand_model_pool = []
            if hand_model_paths is not None:
                from stable_baselines3 import SAC, PPO
                for path in hand_model_paths:
                    try:
                        model = PPO.load(path, custom_objects={'learning_rate': 0.0, 'optimizer_class': None}, verbose=0)
                        self.hand_model_pool.append(model)
                        iteration_dir = os.path.dirname(os.path.dirname(os.path.dirname(path)))
                        self.hand_model_names.append(os.path.basename(iteration_dir))
                    except Exception as e:
                        print(f" {path}, Error: {e}")



        
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
        self.reward_bound = -80
        self.reward_step = -0.2 if training_mode == 'hand' else 0.2
        # self.reward_survival = 10
        
        # Biomechanical Cost Weights
        self.w_sweep = 10.0
        self.w_effort = 2.0
        
        # --- Movement Params ---
        self.stride_robot_random = [0.58, 0.62]
        # dual: [0.3, 0.4]

        self.stride_hand_random = [0.3, 0.5]
        self.scripted_hand_stride_random = [0.4, 0.6]
        self.hand_move_epsilon = 0.05
        
        self.max_steps = 100
        
        self.steps = 0
        
        # --- Spaces ---
        self.action_space = Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observation space definition
        self.opponent_id_dim = 0
        if self.training_mode == 'robot' and self.include_opponent_id:
            self.opponent_id_dim = 1 + len(self.hand_model_pool)
            self.current_opponent_one_hot_dim = self.opponent_id_dim
        # obs dim calculation:
        # actor position(2) + opponent position(2) + dist(1) + bounds(4) + stride(1)
        # + last action(2) + history(16 * 2) + optional opponent id one-hot
        self.obs_dim = obs_dim(self.history_length, self.opponent_id_dim, self.history_channels)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # --- Internals ---
        self.robot_position = np.zeros(2)
        self.hand_position = np.zeros(2)
        self.blocking_point = np.zeros(2)

        # ========================================================
        # []and ?
        # ========================================================
        self.last_hand_actual_move = np.zeros(2, dtype=np.float32)

        # ========================================================
        # []obot ?
        # ========================================================
        self.last_robot_action = np.zeros(2, dtype=np.float32)
        self.last_robot_raw_action = np.zeros(2, dtype=np.float32)
        self.prev_robot_raw_action = np.zeros(2, dtype=np.float32)
        self._jerk_penalty = 0.0

        # ========================================================
        # []lphaStar PFSP ?
        # ========================================================
        self.pfsp_stats = {}        # {pool_index: {'total_steps': 0, 'episodes': 0}}
        self.pfsp_pending_stats = {}  # Batched updates before committing into pfsp_stats
        self.pfsp_update_interval = 20
        self.pfsp_episode_counter = 0
        self.pfsp_last_log_episode = 0
        self.pfsp_temperature = 1.0  # ?
        self.pfsp_min_prob = 0.05   # ?
        self.scripted_hand_sample_prob = 0.2  #  Robot 
        self._current_hand_index = None  # ?pool ?1  scripted hand
        self._episode_pfsp_probs = []
        self._episode_selected_opponent_prob = 1.0
        self._episode_selected_opponent_name = "scripted_hand"
        self._episode_distances = []
        self._episode_bio_costs = []
        self._episode_jerks = []

        # ?hand_position ?
        self._bypass_hand_physics = False

        self.hand_history_buffer = deque(maxlen=self.history_length)
        self.robot_history_buffer = deque(maxlen=self.history_length)
        self.interaction_history_buffer = deque(maxlen=self.history_length)
        self.trajectory_points = []
        
        self.window = None
        self.clock = None
        self.random_noise = True
        self.noise_sigma = 0.05

    def _seed_history_buffer(self, buffer, dim=2):
        buffer.clear()
        for _ in range(self.history_length):
            buffer.append(np.zeros(dim, dtype=np.float32))

    def _compute_pfsp_probabilities(self):
        """Compute PFSP sampling probabilities from average survival length."""
        pool_size = len(self.hand_model_pool)
        scores = np.zeros(pool_size)

        for i in range(pool_size):
            if i in self.pfsp_stats and self.pfsp_stats[i]['episodes'] >= 3:
                avg_len = self.pfsp_stats[i]['total_steps'] / self.pfsp_stats[i]['episodes']
                scores[i] = 1.0 / max(avg_len, 1.0)  # ?
            else:
                scores[i] = 1.0  # 

        # 
        raw_probs = scores / scores.sum() if scores.sum() > 0 else np.ones(pool_size) / pool_size

        # ?
        min_prob = self.pfsp_min_prob / pool_size
        adjusted_probs = np.maximum(raw_probs, min_prob)
        adjusted_probs = adjusted_probs / adjusted_probs.sum()

        return adjusted_probs

    def get_pfsp_stats(self):
        """
         PFSP  debug?
        Returns dict with pool index -> {avg_steps, episodes, probability}
        """
        result = {}
        pool_size = len(self.hand_model_pool)
        probs = self._compute_pfsp_probabilities()

        for i in range(pool_size):
            if i in self.pfsp_stats and self.pfsp_stats[i]['episodes'] > 0:
                avg_steps = self.pfsp_stats[i]['total_steps'] / self.pfsp_stats[i]['episodes']
            else:
                avg_steps = None

            model_name = self.hand_model_names[i] if i < len(self.hand_model_names) else f"hand_{i}"
            result[model_name] = {
                'pool_index': i,
                'episodes': self.pfsp_stats[i]['episodes'] if i in self.pfsp_stats else 0,
                'total_steps': self.pfsp_stats[i]['total_steps'] if i in self.pfsp_stats else 0,
                'avg_episode_steps': avg_steps,
                'selection_prob': probs[i] if i < len(probs) else 0,
            }
        return result

    def _sample_episode_parameters(self):
        # ========================================================
        # PFSP: Priority Fictitious Self-Play ( Robot ?
        # ?Hand ?hand_model
        # AlphaStar ?
        # ========================================================
        self._episode_pfsp_probs = []
        self._episode_selected_opponent_prob = 1.0
        self._episode_selected_opponent_name = "scripted_hand"
        self._current_hand_index = None
        self.current_opponent_id = 0

        if self.training_mode == 'robot' and len(self.hand_model_pool) > 0:
            pool_size = len(self.hand_model_pool)
            scripted_prob = float(np.clip(self.scripted_hand_sample_prob, 0.0, 1.0))
            model_probs = self._compute_pfsp_probabilities() * (1.0 - scripted_prob)
            p = np.concatenate(([scripted_prob], model_probs))
            p = p / p.sum()
            chosen_idx = int(np.random.choice(pool_size + 1, p=p))

            self._episode_pfsp_probs = p.astype(float).tolist()
            self._episode_selected_opponent_prob = float(p[chosen_idx])

            if chosen_idx == 0:
                self.hand_model = None
                self._current_hand_index = -1
                self.current_opponent_id = 0
                self._episode_selected_opponent_name = "scripted_hand"
            else:
                model_idx = chosen_idx - 1
                self.hand_model = self.hand_model_pool[model_idx]
                self._current_hand_index = model_idx
                self.current_opponent_id = model_idx + 1
                if model_idx < len(self.hand_model_names):
                    self._episode_selected_opponent_name = self.hand_model_names[model_idx]
                else:
                    self._episode_selected_opponent_name = f"hand_{model_idx}"

            # Debug:  PFSP 
            # stats = self.get_pfsp_stats()
            # selected_name = list(stats.keys())[chosen_idx] if chosen_idx < len(stats) else str(chosen_idx)
            # print(f"    [PFSP] Selected: {selected_name} (prob={p[chosen_idx]:.3f}) | Stats: {stats}")

        self.stride_robot = np.random.uniform(*self.stride_robot_random)
        self.stride_hand = np.random.uniform(*self.stride_hand_random)
        self.arm_blocking_length = np.random.uniform(0, 1)
        # Scripted hand stride: sampled once per episode, fixed within episode
        self.scripted_hand_stride = random.uniform(*self.scripted_hand_stride_random)

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
        """"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def _calculate_fix_point(self):
        """"""
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
        """ (Hand Agent)"""
        arm_vec = current_pos - self.fixed_point
        arm_len = np.linalg.norm(arm_vec)
        if arm_len < 1e-4: return 0
        arm_dir = arm_vec / arm_len
        v_rad = np.dot(move_vec, arm_dir) * arm_dir # ()
        v_tan = move_vec - v_rad # ()
        s_tan = np.linalg.norm(v_tan)
        s_rad = np.linalg.norm(v_rad)
        #  (Prevent full-screen sweep)
        p_sweep = self.w_sweep * (s_tan**2) * (1 + 2.0 * (arm_len / self.max_length_arm)**2)
        p_effort = self.w_effort * (s_rad**2)
        return p_sweep + p_effort

    def _get_robot_obs(self):
        """Build robot observation."""
        dist_bounds = np.array([
            self.robot_position[0],
            self.env_width - self.robot_position[0],
            self.robot_position[1],
            self.env_height - self.robot_position[1]
        ])
        
        if self.history_mode == "interaction":
            flat_history = np.array(self.interaction_history_buffer, dtype=np.float32).flatten()
            expected_history_dim = self.history_length * self.history_channels
        else:
            flat_history = np.array(self.hand_history_buffer, dtype=np.float32).flatten()
            expected_history_dim = self.history_length * HISTORY_CHANNELS
        if len(flat_history) < expected_history_dim:
            flat_history = np.zeros(expected_history_dim, dtype=np.float32)

        #  APF (Robot ?
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
            self.last_robot_action,  # +2 dims: previous actual robot move
            flat_history,
        )).astype(np.float32)

        if self.include_opponent_id:
            opponent_id = np.zeros(self.opponent_id_dim, dtype=np.float32)
            opponent_id[self.current_opponent_id] = 1.0
            obs = np.concatenate((obs, opponent_id)).astype(np.float32)

        return obs

    def _get_hand_obs(self):
        """Build hand observation."""
        # Hand ?APF ?Robot ??
        # ?History  Robot ?
        
        #  (Hand ?
        dist_bounds = np.array([
            self.hand_position[0],
            self.env_width - self.hand_position[0],
            self.hand_position[1],
            self.env_height - self.hand_position[1]
        ])
        
        flat_robot_history = np.array(self.robot_history_buffer).flatten()
        if len(flat_robot_history) < self.history_length * 2:
            flat_robot_history = np.zeros(self.history_length * 2)

        # Hand ?APF Force 0  Robot 
        # ?32??relative_velocity
        # rel_vel = (self.robot_position - self.hand_position) # ?
        rel_vel = np.zeros(2) # ?

        obs = np.concatenate((
            self.hand_position,       #  (Egocentric)
            self.robot_position,      # ?
            [self.current_distance],
            dist_bounds,
            [self.stride_hand],       # ?
            self.last_hand_actual_move,  # +2 dims: previous actual hand move
            flat_robot_history,       # Hand ?Robot ?
        )).astype(np.float32)
        return obs

    # --- 4.  _get_obs  ---
    def _get_obs(self):
        """ Agent Obs"""
        if self.training_mode == 'robot':
            return self._get_robot_obs()
        else:
            return self._get_hand_obs()

    def _build_league_episode_info(self, reward, terminated, truncated, done_reason):
        distances = np.asarray(self._episode_distances, dtype=float)
        z_min = 4.0
        z_max = 6.0
        if distances.size == 0:
            zpd_coverage = 0.0
            tis = 0.0
            avg_distance = 0.0
            min_distance = 0.0
            max_distance = 0.0
            too_close_rate = 0.0
            too_far_rate = 0.0
        else:
            in_zpd = (distances >= z_min) & (distances <= z_max)
            zpd_coverage = float(np.mean(in_zpd))
            tis = float(np.sum(in_zpd) / max(self.max_steps, 1))
            avg_distance = float(np.mean(distances))
            min_distance = float(np.min(distances))
            max_distance = float(np.max(distances))
            too_close_rate = float(np.mean(distances < z_min))
            too_far_rate = float(np.mean(distances > z_max))

        return {
            "training_mode": self.training_mode,
            "episode_length": int(self.steps),
            "max_steps": int(self.max_steps),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "done_reason": done_reason,
            "zpd_min": z_min,
            "zpd_max": z_max,
            "tis": tis,
            "zpd_coverage": zpd_coverage,
            "too_close_rate": too_close_rate,
            "too_far_rate": too_far_rate,
            "avg_distance": avg_distance,
            "min_distance": min_distance,
            "max_distance": max_distance,
            "mean_bio_cost": float(np.mean(self._episode_bio_costs)) if self._episode_bio_costs else 0.0,
            "mean_jerk": float(np.mean(self._episode_jerks)) if self._episode_jerks else 0.0,
            "selected_opponent_index": 0 if self._current_hand_index == -1 else (None if self._current_hand_index is None else int(self._current_hand_index) + 1),
            "opponent_id": int(self.current_opponent_id),
            "opponent_id_dim": int(self.opponent_id_dim),
            "include_opponent_id": bool(self.include_opponent_id),
            "selected_opponent_name": self._episode_selected_opponent_name,
            "selected_opponent_prob": float(self._episode_selected_opponent_prob),
            "pfsp_probs": list(self._episode_pfsp_probs),
            "pfsp_pool_size": len(self._episode_pfsp_probs),
            "pfsp_update_interval": int(self.pfsp_update_interval),
            "pfsp_pending_episodes": int(sum(s['episodes'] for s in self.pfsp_pending_stats.values())),
        }

    def _get_info(self):
        return {
            "dist": self.current_distance,
            "steps": self.steps,
            "hand_pos": self.hand_position,
            "robot_pos": self.robot_position,
            "fixed_point": self.fixed_point,
            "blocking_point": self.blocking_point,
            "current_hand_index": self._current_hand_index,
            "opponent_id": int(self.current_opponent_id),
            "opponent_id_dim": int(self.opponent_id_dim),
            "include_opponent_id": bool(self.include_opponent_id),
            "selected_opponent_name": self._episode_selected_opponent_name,
            "pfsp_probs": list(self._episode_pfsp_probs),
        }



    def reset_patient(self):
        self.current_patient_param = self.patient.reset_randomly()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._sample_episode_parameters()
        self._sample_initial_positions()
        self._seed_history_buffer(self.hand_history_buffer, HISTORY_CHANNELS)
        self._seed_history_buffer(self.robot_history_buffer, HISTORY_CHANNELS)
        self._seed_history_buffer(self.interaction_history_buffer, self.history_channels)
        self.trajectory_points = [self.robot_position.copy()]
        self.current_distance = np.linalg.norm(self.robot_position - self.hand_position)
        self.pre_distance = self.current_distance
        self.steps = 0

        # []?
        self.last_hand_actual_move = np.zeros(2, dtype=np.float32)
        self.last_robot_action = np.zeros(2, dtype=np.float32)
        self.last_robot_raw_action = np.zeros(2, dtype=np.float32)
        self.prev_robot_raw_action = np.zeros(2, dtype=np.float32)
        self._jerk_penalty = 0.0
        self._bypass_hand_physics = False
        self._episode_distances = []
        self._episode_bio_costs = []
        self._episode_jerks = []

        # Reset biomechanical filter for new episode
        self.biomech_filter.reset()

        self._calculate_fix_point()
        # ?
        self.hand_dir = (self.hand_position - self.fixed_point) / np.linalg.norm(self.hand_position - self.fixed_point)
        self.blocking_point = self.hand_position # blocking point?
        
        return self._get_obs(), self._get_info()

    def _get_scripted_hand_move(self):
        """Scripted hand fallback controller."""
        # Scripted hand stride is fixed within episode, sampled in reset/_sample_episode_parameters
        scripted_stride = self.scripted_hand_stride
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
            expected_obs_dim = model_obs_dim(self.robot_model)
            if expected_obs_dim is not None:
                obs_for_robot = adapt_history_obs(
                    obs_for_robot,
                    target_obs_dim=expected_obs_dim,
                    opponent_id_dim=self.robot_opponent_id_dim,
                )
            robot_action, _ = self.robot_model.predict(obs_for_robot, deterministic=False)
            # Track raw robot action for action_diff penalty
            self.prev_robot_raw_action = self.last_robot_raw_action.copy()
            self.last_robot_raw_action = robot_action.copy()
            return robot_action * self.stride_robot

        # Robot mode: use RL action directly
        self.prev_robot_raw_action = self.last_robot_raw_action.copy()
        self.last_robot_raw_action = action.copy()
        return action * self.stride_robot

    def _resolve_hand_move(self, action):
        # ?hand_position?
        if getattr(self, '_bypass_hand_physics', False):
            return np.zeros(2)

        # 1. "" (Raw Action)
        if self.training_mode == 'hand':
            hand_intent = action * self.stride_hand
        elif self.hand_model is None:
            # print("?Robot  Hand ")
            hand_intent = self._get_scripted_hand_move()
        else:
            obs_for_hand = self._get_hand_obs()
            expected_obs_dim = model_obs_dim(self.hand_model)
            if expected_obs_dim is not None:
                obs_for_hand = adapt_history_obs(obs_for_hand, target_obs_dim=expected_obs_dim)
            hand_action, _ = self.hand_model.predict(obs_for_hand, deterministic=True)
            hand_intent = hand_action * self.stride_hand

        # ========================================================
        # 2.  I?(Muscle Inertia)
        # ========================================================
        alpha = 0.5
        smoothed_move = alpha * hand_intent + (1.0 - alpha) * self.last_hand_actual_move
        # smoothed_move = hand_intent

        # ========================================================
        # 3.  II (Acceleration Clipping)
        # ========================================================
        max_accel = 1
        delta_v = smoothed_move - self.last_hand_actual_move
        accel_magnitude = np.linalg.norm(delta_v)

        if accel_magnitude > max_accel:
            delta_v = (delta_v / accel_magnitude) * max_accel

        final_physics_move = self.last_hand_actual_move + delta_v

        # ========================================================
        # 4. 
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



        dist_improvement = self.pre_distance - self.current_distance
        if self.training_mode == 'robot':
            reward += self.reward_step
            # Robot erk?
            #  Robot ""?
            jerk_penalty = -2 * self._jerk_penalty
            reward += jerk_penalty
            # # Action smoothness penalty: penalize large changes in raw robot action
            # action_diff = np.linalg.norm(self.last_robot_raw_action - self.prev_robot_raw_action)
            # action_diff_penalty = -0.05 * action_diff
            # reward += action_diff_penalty
        else:
            # Hand 
            # 1. ?
            reward += dist_improvement * 0.5
            # 2. ?bonus 
            # max_distance ???18, min_distance = 0
            proximity_reward = max(0, (18 - self.current_distance) / 18) * 0.1
            reward += proximity_reward
            # 3. ?hand ?
            reward += self.reward_step

        self.pre_distance = self.current_distance

        if self.steps >= self.max_steps:
            truncated = True

        # ========================================================
        # [AlphaStar PFSP] 
        # ========================================================
        if (terminated or truncated) and self.training_mode == 'robot' and self._current_hand_index is not None and self._current_hand_index >= 0:
            pool_idx = self._current_hand_index
            if pool_idx not in self.pfsp_pending_stats:
                self.pfsp_pending_stats[pool_idx] = {'total_steps': 0, 'episodes': 0}
            self.pfsp_pending_stats[pool_idx]['total_steps'] += self.steps
            self.pfsp_pending_stats[pool_idx]['episodes'] += 1
            self.pfsp_episode_counter += 1

            if self.pfsp_episode_counter % self.pfsp_update_interval == 0:
                for idx, stats in self.pfsp_pending_stats.items():
                    if idx not in self.pfsp_stats:
                        self.pfsp_stats[idx] = {'total_steps': 0, 'episodes': 0}
                    self.pfsp_stats[idx]['total_steps'] += stats['total_steps']
                    self.pfsp_stats[idx]['episodes'] += stats['episodes']
                self.pfsp_pending_stats.clear()

            # Debug: ?1000 ?episode ?PFSP 
            total_episodes = sum(s['episodes'] for s in self.pfsp_stats.values())
            if total_episodes > 0 and total_episodes % 1000 == 0 and total_episodes != self.pfsp_last_log_episode:
                self.pfsp_last_log_episode = total_episodes
                stats = self.get_pfsp_stats()
                print(f"    [PFSP] Episodes: {total_episodes} | {stats}")

        return reward, terminated, truncated, done_reason

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        bio_cost = 0.0

        if self.random_noise:
            action += np.random.normal(0, self.noise_sigma, size=action.shape)

        previous_distance = float(self.current_distance)
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

        # Robot ?jerk?
        self._jerk_penalty = np.linalg.norm(robot_actual_move - self.last_robot_action)
        # ?
        self.last_robot_action = robot_actual_move.copy()
        
        self._calculate_fix_point()
        vec_arm = self.fixed_point - self.hand_position
        dist_arm = np.linalg.norm(vec_arm)
        to_shoulder = self.safe_normalize(vec_arm)
        self.blocking_point = self.hand_position + to_shoulder * min(self.arm_blocking_length, dist_arm)

        self.hand_dir  = -to_shoulder

        self.steps += 1

        reward, terminated, truncated, done_reason = self._compute_reward_and_done(old_robot_pos)
        relative_pos = self.hand_position - self.robot_position
        interaction_features = np.array([
            relative_pos[0],
            relative_pos[1],
            self.current_distance,
            self.current_distance - previous_distance,
            robot_actual_move[0],
            robot_actual_move[1],
            hand_move_final[0],
            hand_move_final[1],
        ], dtype=np.float32)
        self.interaction_history_buffer.append(interaction_features)
        self._episode_distances.append(float(self.current_distance))
        self._episode_bio_costs.append(float(bio_cost))
        self._episode_jerks.append(float(self._jerk_penalty))

        obs = self._apply_obs_noise(self._get_obs())
        info = self._get_info()
        info['bio_cost'] = bio_cost
        info['done_reason'] = done_reason
        if terminated or truncated:
            info['league_episode'] = self._build_league_episode_info(reward, terminated, truncated, done_reason)

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



