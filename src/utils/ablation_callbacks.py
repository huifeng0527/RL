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

    def __init__(
        self,
        batch_size=256,
        buffer_size=10000,
        aux_epochs=3,
        history_length=DEFAULT_HISTORY_LENGTH,
        history_channels=HISTORY_CHANNELS,
        verbose=0,
    ):
        super().__init__(verbose)
        self.batch_size = batch_size
        self.aux_epochs = aux_epochs
        self.history_length = int(history_length)
        self.history_channels = int(history_channels)
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

                h_slice = history_slice(self.history_length, self.history_channels)
                true_next_move = batch_next_obs_tensor[:, h_slice.stop - HISTORY_CHANNELS:h_slice.stop]
                pred_next_move = self.extractor.forward_aux(batch_obs_tensor)
                loss = F.mse_loss(pred_next_move, true_next_move)

                self.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.extractor.parameters(), max_norm=0.5)
                self.optimizer.step()

                epoch_losses.append(loss.item())

            self.logger.record("auxiliary/prediction_loss", np.mean(epoch_losses))


class PPOFutureAuxTrainingCallback(BaseCallback):
    """Auxiliary training for strategy inference from recent interaction history."""

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
        contrastive_weight=0.05,
        contrastive_temperature=0.1,
        traj_cumulative_weight=0.0,
        traj_endpoint_weight=0.0,
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
        self.contrastive_weight = float(contrastive_weight)
        self.contrastive_temperature = float(contrastive_temperature)
        self.traj_cumulative_weight = float(traj_cumulative_weight)
        self.traj_endpoint_weight = float(traj_endpoint_weight)
        self.catch_threshold = float(catch_threshold)
        self.episodes = deque(maxlen=buffer_size)
        self.current_episodes = None
        self.current_labels = None
        self.optimizer = None

    def _on_training_start(self) -> None:
        self.extractor = self.model.policy.features_extractor
        self.optimizer = th.optim.Adam(self.extractor.parameters(), lr=self.lr)
        self.current_episodes = [[] for _ in range(self.training_env.num_envs)]
        self.current_labels = [None for _ in range(self.training_env.num_envs)]

    def _opponent_label_from_info(self, info):
        if not isinstance(info, dict):
            return None
        if info.get("opponent_id") is not None:
            return int(info["opponent_id"])
        league_episode = info.get("league_episode")
        if isinstance(league_episode, dict) and league_episode.get("opponent_id") is not None:
            return int(league_episode["opponent_id"])
        current_hand_index = info.get("current_hand_index")
        if current_hand_index is not None:
            return int(current_hand_index) + 1
        return None

    def _on_step(self) -> bool:
        obs_array = self.model._last_obs
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", []) or []

        for env_idx in range(obs_array.shape[0]):
            self.current_episodes[env_idx].append(np.array(obs_array[env_idx], dtype=np.float32))
            info = infos[env_idx] if len(infos) > env_idx else {}
            label = self._opponent_label_from_info(info)
            if label is not None:
                self.current_labels[env_idx] = label
            if len(dones) > env_idx and dones[env_idx]:
                episode = self.current_episodes[env_idx]
                episode_label = label if label is not None else self.current_labels[env_idx]
                if len(episode) > self.future_horizon and (episode_label is not None or self.contrastive_weight == 0):
                    self.episodes.append({
                        "obs": np.stack(episode, axis=0),
                        "label": int(episode_label) if episode_label is not None else 0,
                    })
                self.current_episodes[env_idx] = []
                self.current_labels[env_idx] = None
        return True

    def _sample_window(self, episode_record, h_slice, hand_delta_start):
        ep = episode_record["obs"]
        t = np.random.randint(0, len(ep) - self.future_horizon)
        future = ep[t + 1:t + 1 + self.future_horizon]
        future_moves = future[:, hand_delta_start:h_slice.stop]
        min_dist = float(np.min(future[:, 4]))
        risk = 1.0 if min_dist < self.catch_threshold else 0.0
        return ep[t], future_moves, risk, int(episode_record["label"])

    def _sample_batch(self):
        valid = [ep for ep in self.episodes if len(ep["obs"]) > self.future_horizon]
        if not valid:
            return None

        episodes_by_label = {}
        for ep in valid:
            episodes_by_label.setdefault(int(ep["label"]), []).append(ep)
        labels_available = list(episodes_by_label.keys())
        if not labels_available:
            return None

        obs_batch = []
        future_moves = []
        risk_targets = []
        labels = []
        h_slice = history_slice(self.history_length, self.history_channels)
        latest_step_start = h_slice.stop - self.history_channels
        hand_delta_start = latest_step_start + self.history_channels - HISTORY_CHANNELS

        while len(obs_batch) < self.batch_size:
            label = labels_available[np.random.randint(len(labels_available))]
            repeats = min(2 if len(labels_available) >= 2 else 1, self.batch_size - len(obs_batch))
            for _ in range(repeats):
                episode_record = episodes_by_label[label][np.random.randint(len(episodes_by_label[label]))]
                obs, moves, risk, sampled_label = self._sample_window(episode_record, h_slice, hand_delta_start)
                obs_batch.append(obs)
                future_moves.append(moves)
                risk_targets.append(risk)
                labels.append(sampled_label)

        return (
            np.asarray(obs_batch, dtype=np.float32),
            np.asarray(future_moves, dtype=np.float32),
            np.asarray(risk_targets, dtype=np.float32),
            np.asarray(labels, dtype=np.int64),
        )

    def _supervised_contrastive_loss(self, embeddings, labels):
        unique_labels = th.unique(labels)
        if embeddings.shape[0] < 2 or unique_labels.numel() < 2:
            return embeddings.new_tensor(0.0)

        z = F.normalize(embeddings, dim=1)
        logits = th.matmul(z, z.T) / self.contrastive_temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        batch_size = labels.shape[0]
        non_self = ~th.eye(batch_size, dtype=th.bool, device=labels.device)
        positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & non_self
        positive_counts = positive_mask.sum(dim=1)
        valid = positive_counts > 0
        if not th.any(valid):
            return embeddings.new_tensor(0.0)

        log_denom = th.logsumexp(logits.masked_fill(~non_self, float("-inf")), dim=1, keepdim=True)
        log_prob = logits - log_denom
        per_anchor_loss = -(log_prob * positive_mask.float()).sum(dim=1) / positive_counts.clamp_min(1)
        return per_anchor_loss[valid].mean()

    def _on_rollout_end(self) -> None:
        if len(self.episodes) == 0:
            return

        traj_losses = []
        traj_step_losses = []
        traj_cumulative_losses = []
        traj_endpoint_losses = []
        risk_losses = []
        contrastive_losses = []
        total_losses = []

        for _ in range(self.aux_epochs):
            batch = self._sample_batch()
            if batch is None:
                return
            obs_np, future_moves_np, risk_np, labels_np = batch

            device = self.model.device
            obs = th.tensor(obs_np, dtype=th.float32, device=device)
            future_moves = th.tensor(future_moves_np, dtype=th.float32, device=device)
            risk = th.tensor(risk_np, dtype=th.float32, device=device)
            labels = th.tensor(labels_np, dtype=th.long, device=device)

            pred_traj, pred_risk_logit = self.extractor.forward_aux_future(obs)
            traj_step_loss = F.mse_loss(pred_traj, future_moves)
            pred_cumulative = th.cumsum(pred_traj, dim=1)
            true_cumulative = th.cumsum(future_moves, dim=1)
            traj_cumulative_loss = F.mse_loss(pred_cumulative, true_cumulative)
            traj_endpoint_loss = F.mse_loss(pred_cumulative[:, -1], true_cumulative[:, -1])
            traj_loss = (
                traj_step_loss
                + self.traj_cumulative_weight * traj_cumulative_loss
                + self.traj_endpoint_weight * traj_endpoint_loss
            )
            risk_loss = F.binary_cross_entropy_with_logits(pred_risk_logit, risk)
            if self.contrastive_weight > 0:
                strategy_embeddings = self.extractor.forward_strategy_embedding(obs)
                contrastive_loss = self._supervised_contrastive_loss(strategy_embeddings, labels)
            else:
                contrastive_loss = obs.new_tensor(0.0)
            loss = (
                self.traj_weight * traj_loss
                + self.risk_weight * risk_loss
                + self.contrastive_weight * contrastive_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.extractor.parameters(), max_norm=0.5)
            self.optimizer.step()

            traj_losses.append(traj_loss.item())
            traj_step_losses.append(traj_step_loss.item())
            traj_cumulative_losses.append(traj_cumulative_loss.item())
            traj_endpoint_losses.append(traj_endpoint_loss.item())
            risk_losses.append(risk_loss.item())
            contrastive_losses.append(contrastive_loss.item())
            total_losses.append(loss.item())

        self.logger.record("strategy_aux/loss", np.mean(total_losses))
        self.logger.record("strategy_aux/future_traj_loss", np.mean(traj_losses))
        self.logger.record("strategy_aux/future_step_loss", np.mean(traj_step_losses))
        self.logger.record("strategy_aux/future_cumulative_loss", np.mean(traj_cumulative_losses))
        self.logger.record("strategy_aux/future_endpoint_loss", np.mean(traj_endpoint_losses))
        self.logger.record("strategy_aux/catch_risk_loss", np.mean(risk_losses))
        self.logger.record("strategy_aux/contrastive_loss", np.mean(contrastive_losses))
        self.logger.record("strategy_aux/label_count", len({ep["label"] for ep in self.episodes}))


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
