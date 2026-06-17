"""Ablation study callbacks.

Extracts auxiliary training callbacks and metrics saving from ab_study.py.
"""

import os
import numpy as np
import torch as th
import torch.nn.functional as F
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback

from src.observation_schema import (
    HISTORY_CHANNELS,
    INTERACTION_HISTORY_CHANNELS,
    DEFAULT_HISTORY_LENGTH,
    OBS_SCALAR_DIM,
    history_slice,
)

HISTORY_LENGTH = DEFAULT_HISTORY_LENGTH
HISTORY_END = OBS_SCALAR_DIM + HISTORY_LENGTH * HISTORY_CHANNELS


class AuxTrainingCallback(BaseCallback):
    """Auxiliary task training callback for SAC.

    Trains feature extractor to predict next-frame hand displacement.
    """

    def __init__(self, train_freq=1000, batch_size=256, verbose=0):
        super().__init__(verbose)
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.optimizer = None

    def _on_training_start(self) -> None:
        self.extractor = self.model.policy.actor.features_extractor
        self.optimizer = th.optim.Adam(self.extractor.parameters(), lr=5e-5)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.train_freq == 0 and self.model.replay_buffer.size() > self.batch_size:
            replay_data = self.model.replay_buffer.sample(self.batch_size)

            obs = replay_data.observations
            next_obs = replay_data.next_observations

            true_next_move = next_obs[:, HISTORY_END - HISTORY_CHANNELS:HISTORY_END]
            pred_next_move = self.extractor.forward_aux(obs)

            loss = F.mse_loss(pred_next_move, true_next_move)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.logger.record("auxiliary/prediction_loss", loss.item())
        return True


class PPOAuxTrainingCallback(BaseCallback):
    """Auxiliary task training callback for PPO.

    Collects rollout data and trains auxiliary task at end of each rollout.
    """

    def __init__(self, batch_size=256, buffer_size=10000, aux_epochs=3, verbose=0):
        super().__init__(verbose)
        self.batch_size = batch_size
        self.aux_epochs = aux_epochs
        self.optimizer = None

        self.obs_buffer = deque(maxlen=buffer_size)
        self.next_obs_buffer = deque(maxlen=buffer_size)

    def _on_training_start(self) -> None:
        self.extractor = self.model.policy.features_extractor
        self.optimizer = th.optim.Adam(self.extractor.parameters(), lr=5e-5)

    def _on_step(self) -> bool:
        obs_array = self.model._last_obs
        next_obs_array = self.locals["new_obs"]

        for i in range(obs_array.shape[0]):
            self.obs_buffer.append(obs_array[i])
            self.next_obs_buffer.append(next_obs_array[i])
        return True

    def _on_rollout_end(self) -> None:
        if len(self.obs_buffer) >= self.batch_size:
            epoch_losses = []

            for _ in range(self.aux_epochs):
                indices = np.random.choice(len(self.obs_buffer), self.batch_size, replace=False)
                batch_obs = np.array([self.obs_buffer[idx] for idx in indices])
                batch_next_obs = np.array([self.next_obs_buffer[idx] for idx in indices])

                device = self.model.device
                batch_obs_tensor = th.tensor(batch_obs, dtype=th.float32).to(device)
                batch_next_obs_tensor = th.tensor(batch_next_obs, dtype=th.float32).to(device)

                true_next_move = batch_next_obs_tensor[:, HISTORY_END - HISTORY_CHANNELS:HISTORY_END]
                pred_next_move = self.extractor.forward_aux(batch_obs_tensor)
                loss = F.mse_loss(pred_next_move, true_next_move)

                self.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.extractor.parameters(), max_norm=0.5)
                self.optimizer.step()

                epoch_losses.append(loss.item())

            self.logger.record("auxiliary/prediction_loss", np.mean(epoch_losses))


class PPOFutureAuxTrainingCallback(BaseCallback):
    """Auxiliary training for strategy inference from recent interaction history.

    The callback samples windows from completed rollout episodes and trains the
    extractor to predict future hand-motion deltas, future catch risk, and the
    future minimum robot-hand distance.
    """

    def __init__(
        self,
        history_length=32,
        history_channels=INTERACTION_HISTORY_CHANNELS,
        future_horizon=8,
        batch_size=256,
        buffer_size=256,
        aux_epochs=3,
        lr=5e-5,
        traj_weight=1.0,
        risk_weight=0.2,
        min_dist_weight=0.1,
        catch_threshold=2.0,
        verbose=0,
    ):
        super().__init__(verbose)
        self.history_length = int(history_length)
        self.history_channels = int(history_channels)
        self.future_horizon = int(future_horizon)
        self.batch_size = int(batch_size)
        self.aux_epochs = int(aux_epochs)
        self.lr = float(lr)
        self.traj_weight = float(traj_weight)
        self.risk_weight = float(risk_weight)
        self.min_dist_weight = float(min_dist_weight)
        self.catch_threshold = float(catch_threshold)
        self.episodes = deque(maxlen=buffer_size)
        self.current_episodes = None
        self.optimizer = None

    def _on_training_start(self) -> None:
        self.extractor = self.model.policy.features_extractor
        self.optimizer = th.optim.Adam(self.extractor.parameters(), lr=self.lr)
        self.current_episodes = [[] for _ in range(self.training_env.num_envs)]

    def _on_step(self) -> bool:
        obs_array = self.model._last_obs
        dones = self.locals.get("dones", [])

        for env_idx in range(obs_array.shape[0]):
            self.current_episodes[env_idx].append(np.array(obs_array[env_idx], dtype=np.float32))
            if len(dones) > env_idx and dones[env_idx]:
                episode = self.current_episodes[env_idx]
                if len(episode) > self.future_horizon:
                    self.episodes.append(np.stack(episode, axis=0))
                self.current_episodes[env_idx] = []
        return True

    def _sample_batch(self):
        valid = [ep for ep in self.episodes if len(ep) > self.future_horizon]
        if not valid:
            return None

        obs_batch = []
        future_moves = []
        risk_targets = []
        min_dist_targets = []
        h_slice = history_slice(self.history_length, self.history_channels)
        latest_step_start = h_slice.stop - self.history_channels
        hand_delta_start = latest_step_start + self.history_channels - HISTORY_CHANNELS

        for _ in range(self.batch_size):
            ep = valid[np.random.randint(len(valid))]
            t = np.random.randint(0, len(ep) - self.future_horizon)
            future = ep[t + 1:t + 1 + self.future_horizon]
            obs_batch.append(ep[t])
            future_moves.append(future[:, hand_delta_start:h_slice.stop])
            distances = future[:, 4]
            min_dist = float(np.min(distances))
            min_dist_targets.append(min_dist)
            risk_targets.append(1.0 if min_dist < self.catch_threshold else 0.0)

        return (
            np.asarray(obs_batch, dtype=np.float32),
            np.asarray(future_moves, dtype=np.float32),
            np.asarray(risk_targets, dtype=np.float32),
            np.asarray(min_dist_targets, dtype=np.float32),
        )

    def _on_rollout_end(self) -> None:
        if len(self.episodes) == 0:
            return

        traj_losses = []
        risk_losses = []
        min_dist_losses = []
        total_losses = []

        for _ in range(self.aux_epochs):
            batch = self._sample_batch()
            if batch is None:
                return
            obs_np, future_moves_np, risk_np, min_dist_np = batch

            device = self.model.device
            obs = th.tensor(obs_np, dtype=th.float32, device=device)
            future_moves = th.tensor(future_moves_np, dtype=th.float32, device=device)
            risk = th.tensor(risk_np, dtype=th.float32, device=device)
            min_dist = th.tensor(min_dist_np, dtype=th.float32, device=device)

            pred_traj, pred_risk_logit, pred_min_dist = self.extractor.forward_aux_future(obs)
            traj_loss = F.mse_loss(pred_traj, future_moves)
            risk_loss = F.binary_cross_entropy_with_logits(pred_risk_logit, risk)
            min_dist_loss = F.mse_loss(pred_min_dist, min_dist)
            loss = (
                self.traj_weight * traj_loss
                + self.risk_weight * risk_loss
                + self.min_dist_weight * min_dist_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.extractor.parameters(), max_norm=0.5)
            self.optimizer.step()

            traj_losses.append(traj_loss.item())
            risk_losses.append(risk_loss.item())
            min_dist_losses.append(min_dist_loss.item())
            total_losses.append(loss.item())

        self.logger.record("strategy_aux/loss", np.mean(total_losses))
        self.logger.record("strategy_aux/future_traj_loss", np.mean(traj_losses))
        self.logger.record("strategy_aux/catch_risk_loss", np.mean(risk_losses))
        self.logger.record("strategy_aux/min_distance_loss", np.mean(min_dist_losses))


class SaveMetricsCallback(BaseCallback):
    """Save training metrics to npz file."""

    def __init__(self, save_path, zpd_min=4.0, zpd_max=6.0, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.zpd_min = zpd_min
        self.zpd_max = zpd_max
        self.data = {
            "timesteps": [],
            "rewards": [],
            "ep_lengths": [],
            "zpd_coverage": [],
            "avg_distance": [],
            "too_close_rate": [],
            "too_far_rate": [],
        }
        self._episode_distances = None

    def _on_training_start(self) -> None:
        self._episode_distances = [[] for _ in range(self.training_env.num_envs)]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for env_idx, info in enumerate(infos):
            if "dist" in info and self._episode_distances is not None:
                self._episode_distances[env_idx].append(float(info["dist"]))

            if "episode" in info:
                self.data["timesteps"].append(self.num_timesteps)
                self.data["rewards"].append(info["episode"]["r"])
                self.data["ep_lengths"].append(info["episode"]["l"])

                distances = []
                if self._episode_distances is not None:
                    distances = self._episode_distances[env_idx]
                    self._episode_distances[env_idx] = []

                if distances:
                    dist_arr = np.asarray(distances, dtype=np.float32)
                    self.data["zpd_coverage"].append(
                        np.mean((dist_arr >= self.zpd_min) & (dist_arr <= self.zpd_max))
                    )
                    self.data["avg_distance"].append(np.mean(dist_arr))
                    self.data["too_close_rate"].append(np.mean(dist_arr < self.zpd_min))
                    self.data["too_far_rate"].append(np.mean(dist_arr > self.zpd_max))
                else:
                    self.data["zpd_coverage"].append(np.nan)
                    self.data["avg_distance"].append(np.nan)
                    self.data["too_close_rate"].append(np.nan)
                    self.data["too_far_rate"].append(np.nan)

        return True

    def _on_training_end(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

        np.savez(
            os.path.join(self.save_path, "metrics.npz"),
            timesteps=np.array(self.data["timesteps"]),
            rewards=np.array(self.data["rewards"]),
            ep_lengths=np.array(self.data["ep_lengths"]),
            zpd_coverage=np.array(self.data["zpd_coverage"]),
            avg_distance=np.array(self.data["avg_distance"]),
            too_close_rate=np.array(self.data["too_close_rate"]),
            too_far_rate=np.array(self.data["too_far_rate"]),
        )

        print(f"Metrics saved to {self.save_path}/metrics.npz")
