"""Feature extractors for rehabilitation RL environment.

Extracts from ab_study.py - these are used for ablation studies.
"""

import torch as th
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from ..custom_env import HISTORY_CHANNELS, HISTORY_LENGTH, OBS_SCALAR_DIM


SCALAR_DIM = OBS_SCALAR_DIM
HISTORY_LEN = HISTORY_LENGTH


class MLPOnlyExtractor(BaseFeaturesExtractor):
    """Pure MLP without history - baseline short-sighted model."""

    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=64)
        self.net = nn.Sequential(
            nn.Linear(SCALAR_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        scalar_part = observations[:, :SCALAR_DIM]
        return self.net(scalar_part)


class BiResidualGatedExtractor(BaseFeaturesExtractor):
    """Bidirectional Residual Gated Fusion.

    Solves shortcut learning and modal collapse problems in multi-modal RL.
    """

    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=64)

        self.history_channels = HISTORY_CHANNELS
        self.history_len = HISTORY_LEN
        self.scalar_dim = SCALAR_DIM
        self.hidden_dim = 32

        # Spatial stream
        self.spatial_net = nn.Sequential(
            nn.Linear(self.scalar_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )

        # Temporal stream (LSTM)
        self.lstm_net = nn.LSTM(
            input_size=self.history_channels,
            hidden_size=self.hidden_dim,
            batch_first=True
        )
        self.temporal_ln = nn.LayerNorm(self.hidden_dim)

        # Bidirectional gates
        self.gate_s2t = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )

        self.gate_t2s = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )

        # Fusion
        self.fusion_net = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 64),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        scalar_part = observations[:, :self.scalar_dim]
        history_part = observations[:, self.scalar_dim:]

        spatial_feats = self.spatial_net(scalar_part)

        history_reshaped = history_part.view(-1, self.history_channels, self.history_len).permute(0, 2, 1)
        lstm_out, _ = self.lstm_net(history_reshaped)
        temporal_feats = self.temporal_ln(lstm_out[:, -1, :])

        gate_temporal = self.gate_s2t(spatial_feats)
        gate_spatial = self.gate_t2s(temporal_feats)

        modulated_temporal = temporal_feats * (1.0 + gate_temporal)
        modulated_spatial = spatial_feats * (1.0 + gate_spatial)

        combined = th.cat([modulated_spatial, modulated_temporal], dim=1)
        return self.fusion_net(combined)


class FiLMFeatureExtractor(BaseFeaturesExtractor):
    """Feature-wise Linear Modulation (FiLM).

    Uses LSTM for temporal features and predicts gamma/beta modulation.
    """

    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=64)
        self.scalar_net = nn.Sequential(
            nn.Linear(SCALAR_DIM, 32),
            nn.ReLU(),
        )
        self.lstm_net = nn.LSTM(
            input_size=HISTORY_CHANNELS,
            hidden_size=32,
            batch_first=True,
        )
        self.gamma_head = nn.Linear(32, 32)
        self.beta_head = nn.Linear(32, 32)
        self.output_net = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        scalar_part = observations[:, :SCALAR_DIM]
        history_part = observations[:, SCALAR_DIM:]
        history_input = history_part.reshape(-1, HISTORY_CHANNELS, HISTORY_LEN).permute(0, 2, 1)

        scalar_features = self.scalar_net(scalar_part)
        lstm_out, _ = self.lstm_net(history_input)
        temporal_features = lstm_out[:, -1, :]

        gamma = 1.0 + th.tanh(self.gamma_head(temporal_features))
        beta = self.beta_head(temporal_features)
        fused = scalar_features * gamma + beta
        return self.output_net(fused)


class LSTMExtractor(BaseFeaturesExtractor):
    """MLP + LSTM with history memory."""

    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=64)

        self.scalar_net = nn.Sequential(
            nn.Linear(SCALAR_DIM, 32),
            nn.ReLU()
        )

        self.lstm_net = nn.LSTM(
            input_size=HISTORY_CHANNELS,
            hidden_size=32,
            num_layers=2,
            batch_first=True
        )

        self.fusion_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        scalar_part = observations[:, :SCALAR_DIM]
        history_part = observations[:, SCALAR_DIM:]

        history_input = history_part.reshape(-1, HISTORY_CHANNELS, HISTORY_LEN).permute(0, 2, 1)

        lstm_out, _ = self.lstm_net(history_input)
        lstm_features = lstm_out[:, -1, :]

        scalar_features = self.scalar_net(scalar_part)
        combined = th.cat([lstm_features, scalar_features], dim=1)

        return self.fusion_net(combined)


class AuxLSTMExtractor(LSTMExtractor):
    """MLP + LSTM + Auxiliary prediction head.

    Predicts next-frame hand displacement as auxiliary task.
    """

    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space)

        self.aux_head = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

    def forward_aux(self, observations: th.Tensor) -> th.Tensor:
        """Forward pass for auxiliary task."""
        history_part = observations[:, SCALAR_DIM:]
        history_input = history_part.view(-1, HISTORY_CHANNELS, HISTORY_LEN).permute(0, 2, 1)

        lstm_out, _ = self.lstm_net(history_input)
        lstm_features = lstm_out[:, -1, :]

        return self.aux_head(lstm_features)


EXTRACTOR_REGISTRY = {
    'MLPOnly': MLPOnlyExtractor,
    'BiResidualGated': BiResidualGatedExtractor,
    'FiLM': FiLMFeatureExtractor,
    'LSTM': LSTMExtractor,
    'AuxLSTM': AuxLSTMExtractor,
}
