from __future__ import annotations

from collections import deque
from itertools import product
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    from src.custom_env import RehabilitationEnv


class ConstantVelocityMPCController:
    """Nonlearning receding-horizon baseline with constant-velocity hand prediction.

    The controller uses only current and past observed positions. It does not query
    the scripted hand controller, learned hand model, environment step function, or
    any future hand trajectory.
    """

    def __init__(
        self,
        horizon: int = 5,
        velocity_window: int = 4,
        action_grid: Iterable[float] | None = None,
        discount: float = 0.95,
        boundary_guard: float = 0.20,
        effort_weight: float = 0.02,
        smoothness_weight: float = 0.05,
        collision_penalty: float = 40.0,
        oob_penalty: float = 80.0,
    ):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if velocity_window < 1:
            raise ValueError("velocity_window must be positive")
        self.horizon = int(horizon)
        self.velocity_window = int(velocity_window)
        self.action_grid = np.asarray(
            list(action_grid) if action_grid is not None else [-1.0, -0.5, 0.0, 0.5, 1.0],
            dtype=np.float32,
        )
        if self.action_grid.ndim != 1 or self.action_grid.size == 0:
            raise ValueError("action_grid must contain at least one scalar value")
        self.discount = float(discount)
        self.boundary_guard = float(boundary_guard)
        self.effort_weight = float(effort_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.collision_penalty = float(collision_penalty)
        self.oob_penalty = float(oob_penalty)

        self.candidate_actions = np.asarray(
            list(product(self.action_grid, self.action_grid)),
            dtype=np.float32,
        )
        self.hand_positions: deque[np.ndarray] = deque(maxlen=max(self.velocity_window + 1, 2))
        self.previous_action = np.zeros(2, dtype=np.float32)
        self._last_seen_step: int | None = None

    def reset(self):
        self.hand_positions.clear()
        self.previous_action[:] = 0.0
        self._last_seen_step = None

    @staticmethod
    def _clip_norm(vec: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm > max_norm and norm > 1e-8:
            return (vec / norm * max_norm).astype(np.float32)
        return vec.astype(np.float32)

    def _record_observation(self, env: RehabilitationEnv):
        step = int(getattr(env, "steps", 0))
        if self._last_seen_step != step or not self.hand_positions:
            self.hand_positions.append(np.asarray(env.hand_position, dtype=np.float32).copy())
            self._last_seen_step = step

    def _estimate_hand_velocity(self) -> np.ndarray:
        if len(self.hand_positions) < 2:
            return np.zeros(2, dtype=np.float32)
        positions = np.asarray(self.hand_positions, dtype=np.float32)
        deltas = positions[1:] - positions[:-1]
        if deltas.size == 0:
            return np.zeros(2, dtype=np.float32)
        return np.mean(deltas[-self.velocity_window :], axis=0).astype(np.float32)

    def _clip_to_safe_next_position(
        self,
        env: RehabilitationEnv,
        action: np.ndarray,
    ) -> np.ndarray:
        robot = np.asarray(env.robot_position, dtype=np.float32)
        stride = max(float(getattr(env, "stride_robot", 0.0)), 1e-6)
        low = np.array([env.margin, env.margin], dtype=np.float32) + self.boundary_guard
        high = (
            np.array([env.env_width - env.margin, env.env_height - env.margin], dtype=np.float32)
            - self.boundary_guard
        )
        min_action = (low - robot) / stride
        max_action = (high - robot) / stride
        return np.clip(action, min_action, max_action).astype(np.float32)

    @staticmethod
    def _zpd_reward(distance: float, z_min: float, z_max: float) -> float:
        z_center = 0.5 * (z_min + z_max)
        z_half_width = 0.5 * (z_max - z_min)
        if distance < z_min:
            reward = -0.25 * np.exp(1.4 * (z_min - distance))
        elif distance <= z_max:
            center_score = 1.0 - abs(distance - z_center) / max(z_half_width, 1e-6)
            reward = 0.75 + 0.25 * center_score
        else:
            reward = -0.35 * (distance - z_max)
        return float(np.clip(reward, -1.0, 1.0))

    @staticmethod
    def _robot_out_of_bounds(robot_position: np.ndarray, env: RehabilitationEnv) -> bool:
        return bool(
            np.any(robot_position <= float(env.margin))
            or robot_position[0] >= float(env.env_width - env.margin)
            or robot_position[1] >= float(env.env_height - env.margin)
        )

    def _score_action(
        self,
        env: RehabilitationEnv,
        action: np.ndarray,
        hand_velocity: np.ndarray,
    ) -> tuple[float, int, int, float, float]:
        stride_robot = float(env.stride_robot)
        robot_position = np.asarray(env.robot_position, dtype=np.float32).copy()
        hand_position = np.asarray(env.hand_position, dtype=np.float32).copy()
        hand_low = np.array([env.margin, env.margin], dtype=np.float32)
        hand_high = np.array([env.env_width - env.margin, env.env_height - env.margin], dtype=np.float32)
        robot_move = action.astype(np.float32) * stride_robot
        previous_robot_move = np.asarray(getattr(env, "last_robot_action", np.zeros(2)), dtype=np.float32).copy()

        score = 0.0
        oob_count = 0
        collision_count = 0

        for horizon_index in range(self.horizon):
            robot_position = robot_position + robot_move
            hand_position = np.clip(hand_position + hand_velocity, hand_low, hand_high)
            distance = float(np.linalg.norm(robot_position - hand_position))

            stage_score = self._zpd_reward(distance, float(env.zpd_min), float(env.zpd_max))
            stage_score += float(getattr(env, "reward_step", 0.2))

            jerk = float(np.linalg.norm(robot_move - previous_robot_move))
            stage_score -= self.smoothness_weight * jerk
            stage_score -= self.effort_weight * float(np.dot(action, action))

            if distance < float(env.distance_threshold_collision):
                stage_score -= self.collision_penalty
                collision_count += 1

            if self._robot_out_of_bounds(robot_position, env):
                stage_score -= self.oob_penalty
                oob_count += 1

            score += (self.discount ** horizon_index) * stage_score
            previous_robot_move = robot_move

        action_magnitude = float(np.linalg.norm(action))
        action_change = float(np.linalg.norm(action - self.previous_action))
        return score, oob_count, collision_count, action_magnitude, action_change

    def predict(self, env: RehabilitationEnv) -> np.ndarray:
        self._record_observation(env)
        hand_velocity = self._estimate_hand_velocity()

        best_action = np.zeros(2, dtype=np.float32)
        best_key: tuple[float, int, int, float, float] | None = None

        for raw_action in self.candidate_actions:
            action = np.clip(raw_action, -1.0, 1.0).astype(np.float32)
            action = self._clip_to_safe_next_position(env, action)
            action = np.clip(action, -1.0, 1.0).astype(np.float32)
            score, oob_count, collision_count, action_magnitude, action_change = self._score_action(
                env,
                action,
                hand_velocity,
            )
            key = (
                score,
                -oob_count,
                -collision_count,
                -action_magnitude,
                -action_change,
            )
            if best_key is None or key > best_key:
                best_key = key
                best_action = action.copy()

        best_action = self._clip_to_safe_next_position(env, best_action)
        best_action = np.clip(best_action, -1.0, 1.0).astype(np.float32)
        self.previous_action = best_action.copy()
        return best_action
