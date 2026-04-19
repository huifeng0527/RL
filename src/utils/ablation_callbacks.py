"""Ablation study callbacks.

Extracts auxiliary training callbacks and metrics saving from ab_study.py.
"""

import os
import numpy as np
import torch as th
import torch.nn.functional as F
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


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

            true_next_move = next_obs[:, -2:]
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

                true_next_move = batch_next_obs_tensor[:, -2:]
                pred_next_move = self.extractor.forward_aux(batch_obs_tensor)
                loss = F.mse_loss(pred_next_move, true_next_move)

                self.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.extractor.parameters(), max_norm=0.5)
                self.optimizer.step()

                epoch_losses.append(loss.item())

            self.logger.record("auxiliary/prediction_loss", np.mean(epoch_losses))


class SaveMetricsCallback(BaseCallback):
    """Save training metrics to npz file."""

    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.data = {
            "timesteps": [],
            "rewards": [],
            "ep_lengths": []
        }

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if "episode" in info:
                self.data["timesteps"].append(self.num_timesteps)
                self.data["rewards"].append(info["episode"]["r"])
                self.data["ep_lengths"].append(info["episode"]["l"])

        return True

    def _on_training_end(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

        np.savez(
            os.path.join(self.save_path, "metrics.npz"),
            timesteps=np.array(self.data["timesteps"]),
            rewards=np.array(self.data["rewards"]),
            ep_lengths=np.array(self.data["ep_lengths"])
        )

        print(f"Metrics saved to {self.save_path}/metrics.npz")
