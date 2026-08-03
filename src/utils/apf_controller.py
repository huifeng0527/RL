from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.custom_env import RehabilitationEnv


class ReactiveAPFController:
    def __init__(
        self,
        radial_gain: float = 1.20,
        tangent_gain: float = 0.30,
        boundary_gain: float = 1.35,
        boundary_band: float = 1.25,
        boundary_guard: float = 0.20,
        smoothing: float = 0.10,
    ):
        self.radial_gain = float(radial_gain)
        self.tangent_gain = float(tangent_gain)
        self.boundary_gain = float(boundary_gain)
        self.boundary_band = float(boundary_band)
        self.boundary_guard = float(boundary_guard)
        self.smoothing = float(smoothing)
        self.previous_action = np.zeros(2, dtype=np.float32)

    def reset(self):
        self.previous_action[:] = 0.0

    @staticmethod
    def _safe_normalize(vec: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm < 1e-8:
            return np.zeros_like(vec, dtype=np.float32)
        return (vec / norm).astype(np.float32)

    @staticmethod
    def _clip_norm(vec: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm > max_norm and norm > 1e-8:
            return (vec / norm * max_norm).astype(np.float32)
        return vec.astype(np.float32)

    def _clip_to_safe_next_position(
        self,
        env: RehabilitationEnv,
        action: np.ndarray,
    ) -> np.ndarray:
        robot = np.asarray(env.robot_position, dtype=np.float32)
        stride = max(float(getattr(env, "stride_robot", 0.0)), 1e-6)
        low = np.array([env.margin, env.margin], dtype=np.float32) + self.boundary_guard
        high = (
            np.array(
                [env.env_width - env.margin, env.env_height - env.margin],
                dtype=np.float32,
            )
            - self.boundary_guard
        )
        min_action = (low - robot) / stride
        max_action = (high - robot) / stride
        return np.clip(action, min_action, max_action).astype(np.float32)

    def predict(self, env: RehabilitationEnv) -> np.ndarray:
        robot = np.asarray(env.robot_position, dtype=np.float32)
        hand = np.asarray(env.hand_position, dtype=np.float32)
        away = robot - hand
        distance = float(np.linalg.norm(away))
        away_dir = self._safe_normalize(away)

        z_min = float(env.zpd_min)
        z_max = float(env.zpd_max)
        if distance < z_min:
            radial_strength = 1.15 + 0.35 * (z_min - distance) / max(z_min, 1e-6)
        elif distance <= z_max:
            zone_fraction = (distance - z_min) / max(z_max - z_min, 1e-6)
            radial_strength = 0.85 - 0.45 * np.clip(zone_fraction, 0.0, 1.0)
        else:
            radial_strength = -0.35 * min(
                (distance - z_max) / max(z_max, 1e-6),
                1.0,
            )
        radial = (
            self.radial_gain
            * np.clip(radial_strength, -0.45, 1.35)
            * away_dir
        )

        tangent = (
            np.array([-away_dir[1], away_dir[0]], dtype=np.float32)
            * self.tangent_gain
        )

        x, y = float(robot[0]), float(robot[1])
        left = x - float(env.margin)
        right = float(env.env_width - env.margin) - x
        bottom = y - float(env.margin)
        top = float(env.env_height - env.margin) - y
        boundary = np.zeros(2, dtype=np.float32)
        band = max(self.boundary_band, 1e-6)
        if left < band:
            boundary[0] += (band - left) / band
        if right < band:
            boundary[0] -= (band - right) / band
        if bottom < band:
            boundary[1] += (band - bottom) / band
        if top < band:
            boundary[1] -= (band - top) / band
        boundary *= self.boundary_gain

        raw_action = self._clip_norm(radial + tangent + boundary, 1.0)
        action = (
            self.smoothing * self.previous_action
            + (1.0 - self.smoothing) * raw_action
        )
        action = self._clip_norm(action, 1.0)
        action = self._clip_to_safe_next_position(env, action)
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.previous_action = action.copy()
        return action
