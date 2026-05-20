"""
Trajectory Debug Callback for Real Deployment
===============================================
记录一回合（episode）下 hand 和 robot 的移动轨迹，
以 robot 控制频率（25Hz）为准，每次 step 对应一个时间步。

用法:
    callback = TrajectoryDebugCallback(save_dir="debug_trajectories")
    callback.reset()                              # episode 开始时调用
    callback.record_step(robot_pos, hand_pos)    # 每个控制周期调用
    callback.save_episode()                       # episode 结束时调用
"""

import os
import numpy as np
import json
import time
from pathlib import Path


class TrajectoryDebugCallback:
    """
    记录一回合内 hand 和 robot 的移动轨迹。

    以 robot 控制频率（25Hz）为准，一次 record_step() 调用对应仿真中一个时间步（0.04s）。
    """

    def __init__(self, save_dir: str = "debug_trajectories", episode_id: str = None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if episode_id is None:
            episode_id = time.strftime("%Y%m%d_%H%M%S")
        self.episode_id = episode_id

        # 轨迹数据
        self.robot_positions = []   # [np.array([x, y]), ...]
        self.hand_positions = []    # [np.array([x, y]), ...]
        self.timestamps = []        # [float, ...] 相对于 episode 开始的时间（秒）
        self.step_count = 0
        self.episode_start_time = None

        # 元数据
        self.meta = {
            "episode_id": self.episode_id,
            "control_freq_hz": 25,
            "dt_seconds": 1.0 / 25,
            "environment_bounds": {"w_env": 15, "h_env": 10},
        }

    def reset(self):
        """episode 开始时调用，重置轨迹记录"""
        self.robot_positions = []
        self.hand_positions = []
        self.timestamps = []
        self.step_count = 0
        self.episode_start_time = time.perf_counter()

    def record_step(self, robot_pos, hand_pos):
        """
        每个控制周期调用，记录当前 step 的位置

        Args:
            robot_pos: np.ndarray [2], robot 环境坐标 (x, y)
            hand_pos: np.ndarray [2], hand 环境坐标 (x, y)
        """
        if self.episode_start_time is None:
            self.episode_start_time = time.perf_counter()

        t = time.perf_counter() - self.episode_start_time

        self.robot_positions.append(np.array(robot_pos, dtype=np.float32))
        self.hand_positions.append(np.array(hand_pos, dtype=np.float32))
        self.timestamps.append(t)
        self.step_count += 1

    def save_episode(self, extra_info: dict = None):
        """
        episode 结束时调用，保存轨迹数据到文件

        Args:
            extra_info: dict, 额外信息（如 reward, done_reason 等）
        """
        if len(self.robot_positions) == 0:
            print(f"[TrajectoryDebugCallback] Warning: No data recorded for episode {self.episode_id}")
            return

        # 转换为 numpy 数组
        robot_arr = np.array(self.robot_positions)   # (N, 2)
        hand_arr = np.array(self.hand_positions)       # (N, 2)
        timestamps = np.array(self.timestamps)         # (N,)

        # 计算速度（每 step 的位移）
        robot_vels = np.diff(robot_arr, axis=0) / self.meta["dt_seconds"]  # (N-1, 2)
        hand_vels = np.diff(hand_arr, axis=0) / self.meta["dt_seconds"]    # (N-1, 2)

        # 计算距离
        distances = np.linalg.norm(robot_arr - hand_arr, axis=1)  # (N,)

        data = {
            "episode_id": self.episode_id,
            "meta": self.meta,
            "total_steps": self.step_count,
            "duration_seconds": float(timestamps[-1]) if len(timestamps) > 0 else 0.0,
            "timestamps": timestamps.tolist(),
            "robot_positions": robot_arr.tolist(),
            "hand_positions": hand_arr.tolist(),
            "robot_vels_m_per_s": robot_vels.tolist() if len(robot_vels) > 0 else [],
            "hand_vels_m_per_s": hand_vels.tolist() if len(hand_vels) > 0 else [],
            "distances": distances.tolist(),
            "extra_info": extra_info or {},
        }

        # 保存 JSON
        json_path = self.save_dir / f"episode_{self.episode_id}.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[TrajectoryDebugCallback] Saved: {json_path}")
        print(f"  -> Steps: {self.step_count}, Duration: {data['duration_seconds']:.2f}s")

        # 同时保存为 .npz 格式（方便后续分析）
        npz_path = self.save_dir / f"episode_{self.episode_id}.npz"
        np.savez(npz_path, **{
            "timestamps": timestamps,
            "robot_positions": robot_arr,
            "hand_positions": hand_arr,
            "robot_vels": robot_vels,
            "hand_vels": hand_vels,
            "distances": distances,
        })
        print(f"[TrajectoryDebugCallback] Saved: {npz_path}")

        return str(json_path), str(npz_path)

    def get_replay_data(self):
        """
        返回可用于回放的 numpy 数组字典

        Returns:
            dict with keys: timestamps, robot_positions, hand_positions
        """
        return {
            "timestamps": np.array(self.timestamps),
            "robot_positions": np.array(self.robot_positions),
            "hand_positions": np.array(self.hand_positions),
        }