from __future__ import annotations

from collections import deque
from itertools import product
from typing import TYPE_CHECKING, Iterable

import numpy as np

if TYPE_CHECKING:
    from src.custom_env import RehabilitationEnv


class ConstantVelocityMPCController:
    """Nonlearning sequence MPC with constant-velocity hand prediction.

    The controller uses only current and past observed positions. It does not query
    the scripted hand controller, learned hand model, environment step function, or
    any future hand trajectory. All ordered pairs of candidate actions are scored:
    the first action is applied for one predicted step and the second is held over
    the remaining horizon. Only the first action is executed before replanning.
    """

    def __init__(
        self,
        horizon: int = 5,
        velocity_window: int = 3,
        action_grid: Iterable[float] | None = None,
        discount: float = 0.95,
        effort_weight: float = 0.01,
        smoothness_weight: float = 0.03,
        collision_penalty: float = 120.0,
        oob_penalty: float = 240.0,
        boundary_band: float = 1.0,
        boundary_barrier_weight: float = 16.0,
        collision_buffer: float = 1.0,
        collision_barrier_weight: float = 8.0,
    ):
        if horizon < 2:
            raise ValueError("horizon must be at least 2 for two-move sequence MPC")
        if velocity_window < 1:
            raise ValueError("velocity_window must be positive")
        if not 0.0 < discount <= 1.0:
            raise ValueError("discount must be in (0, 1]")
        if boundary_band <= 0.0:
            raise ValueError("boundary_band must be positive")
        if collision_buffer <= 0.0:
            raise ValueError("collision_buffer must be positive")

        self.horizon = int(horizon)
        self.velocity_window = int(velocity_window)
        self.action_grid = np.asarray(
            list(action_grid) if action_grid is not None else [-1.0, -0.5, 0.0, 0.5, 1.0],
            dtype=np.float32,
        )
        if self.action_grid.ndim != 1 or self.action_grid.size == 0:
            raise ValueError("action_grid must contain at least one scalar value")

        self.discount = float(discount)
        self.effort_weight = float(effort_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.collision_penalty = float(collision_penalty)
        self.oob_penalty = float(oob_penalty)
        self.boundary_band = float(boundary_band)
        self.boundary_barrier_weight = float(boundary_barrier_weight)
        self.collision_buffer = float(collision_buffer)
        self.collision_barrier_weight = float(collision_barrier_weight)

        self.candidate_actions = np.clip(
            np.asarray(
                list(product(self.action_grid, self.action_grid)),
                dtype=np.float32,
            ),
            -1.0,
            1.0,
        ).astype(np.float32)
        action_count = self.candidate_actions.shape[0]
        self.first_actions = np.repeat(self.candidate_actions, action_count, axis=0)
        self.second_actions = np.tile(self.candidate_actions, (action_count, 1))
        self.num_sequences = int(self.first_actions.shape[0])

        self.hand_positions: deque[np.ndarray] = deque(maxlen=max(self.velocity_window + 1, 2))
        self.previous_action = np.zeros(2, dtype=np.float32)
        self._last_seen_step: int | None = None
        self.last_plan_diagnostics: dict[str, object] = {}

    def reset(self):
        self.hand_positions.clear()
        self.previous_action[:] = 0.0
        self._last_seen_step = None
        self.last_plan_diagnostics = {}

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

    @staticmethod
    def _zpd_reward_vectorized(distances: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
        z_center = 0.5 * (z_min + z_max)
        z_half_width = 0.5 * (z_max - z_min)
        rewards = np.empty_like(distances, dtype=np.float32)

        too_close = distances < z_min
        in_zpd = (distances >= z_min) & (distances <= z_max)
        too_far = distances > z_max

        rewards[too_close] = -0.25 * np.exp(1.4 * (z_min - distances[too_close]))
        rewards[in_zpd] = 0.75 + 0.25 * (
            1.0 - np.abs(distances[in_zpd] - z_center) / max(z_half_width, 1e-6)
        )
        rewards[too_far] = -0.35 * (distances[too_far] - z_max)
        return np.clip(rewards, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _boundary_clearance(robot_positions: np.ndarray, env: RehabilitationEnv) -> np.ndarray:
        margin = float(env.margin)
        clearances = np.stack(
            (
                robot_positions[:, 0] - margin,
                float(env.env_width - margin) - robot_positions[:, 0],
                robot_positions[:, 1] - margin,
                float(env.env_height - margin) - robot_positions[:, 1],
            ),
            axis=1,
        )
        return np.min(clearances, axis=1).astype(np.float32)

    def _score_action_sequences(
        self,
        env: RehabilitationEnv,
        hand_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        sequence_count = self.num_sequences
        robot_positions = np.repeat(
            np.asarray(env.robot_position, dtype=np.float32)[None, :],
            sequence_count,
            axis=0,
        )
        hand_position = np.asarray(env.hand_position, dtype=np.float32).copy()
        hand_low = np.array([env.margin, env.margin], dtype=np.float32)
        hand_high = np.array(
            [env.env_width - env.margin, env.env_height - env.margin],
            dtype=np.float32,
        )
        previous_moves = np.repeat(
            np.asarray(getattr(env, "last_robot_action", np.zeros(2)), dtype=np.float32)[None, :],
            sequence_count,
            axis=0,
        )

        scores = np.zeros(sequence_count, dtype=np.float32)
        active = np.ones(sequence_count, dtype=bool)
        terminal_steps = np.full(sequence_count, self.horizon, dtype=np.int32)
        min_clearances = np.full(sequence_count, np.inf, dtype=np.float32)
        min_distances = np.full(sequence_count, np.inf, dtype=np.float32)

        stride_robot = float(env.stride_robot)
        z_min = float(env.zpd_min)
        z_max = float(env.zpd_max)
        collision_threshold = float(env.distance_threshold_collision)
        step_reward = float(getattr(env, "reward_step", 0.2))

        for horizon_index in range(self.horizon):
            was_active = active.copy()
            if not np.any(was_active):
                break

            actions = self.first_actions if horizon_index == 0 else self.second_actions
            robot_moves = actions * stride_robot
            robot_positions = robot_positions + robot_moves
            hand_position = np.clip(hand_position + hand_velocity, hand_low, hand_high)

            distances = np.linalg.norm(robot_positions - hand_position[None, :], axis=1).astype(np.float32)
            clearances = self._boundary_clearance(robot_positions, env)
            min_distances[was_active] = np.minimum(min_distances[was_active], distances[was_active])
            min_clearances[was_active] = np.minimum(min_clearances[was_active], clearances[was_active])

            stage_scores = self._zpd_reward_vectorized(distances, z_min, z_max) + step_reward
            jerks = np.linalg.norm(robot_moves - previous_moves, axis=1)
            efforts = np.sum(actions * actions, axis=1)
            stage_scores -= self.smoothness_weight * jerks
            stage_scores -= self.effort_weight * efforts

            boundary_violation = np.clip(
                (self.boundary_band - clearances) / self.boundary_band,
                0.0,
                1.0,
            )
            stage_scores -= self.boundary_barrier_weight * boundary_violation * boundary_violation

            collision_clearance = distances - collision_threshold
            collision_violation = np.clip(
                (self.collision_buffer - collision_clearance) / self.collision_buffer,
                0.0,
                1.0,
            )
            stage_scores -= self.collision_barrier_weight * collision_violation * collision_violation

            collisions = distances < collision_threshold
            out_of_bounds = clearances <= 0.0
            stage_scores[collisions] -= self.collision_penalty
            stage_scores[out_of_bounds] -= self.oob_penalty

            scores[was_active] += (
                (self.discount ** horizon_index) * stage_scores[was_active]
            ).astype(np.float32)

            newly_terminal = was_active & (collisions | out_of_bounds)
            terminal_steps[newly_terminal] = horizon_index
            active[newly_terminal] = False
            previous_moves = robot_moves

        return scores, terminal_steps, min_clearances, min_distances

    def predict(self, env: RehabilitationEnv) -> np.ndarray:
        self._record_observation(env)
        hand_velocity = self._estimate_hand_velocity()
        scores, terminal_steps, min_clearances, min_distances = self._score_action_sequences(
            env,
            hand_velocity,
        )

        first_action_norms = np.linalg.norm(self.first_actions, axis=1)
        first_action_changes = np.linalg.norm(self.first_actions - self.previous_action[None, :], axis=1)
        best_index = max(
            range(self.num_sequences),
            key=lambda index: (
                float(scores[index]),
                int(terminal_steps[index]),
                float(min_clearances[index]),
                -float(first_action_norms[index]),
                -float(first_action_changes[index]),
            ),
        )

        best_action = np.clip(self.first_actions[best_index], -1.0, 1.0).astype(np.float32)
        self.previous_action = best_action.copy()
        self.last_plan_diagnostics = {
            "score": float(scores[best_index]),
            "terminal_step": int(terminal_steps[best_index]),
            "predicted_terminal": bool(terminal_steps[best_index] < self.horizon),
            "min_boundary_clearance": float(min_clearances[best_index]),
            "min_hand_distance": float(min_distances[best_index]),
            "first_action": best_action.tolist(),
            "planned_second_action": self.second_actions[best_index].astype(float).tolist(),
            "hand_velocity": hand_velocity.astype(float).tolist(),
            "num_sequences": self.num_sequences,
            "horizon": self.horizon,
        }
        return best_action
