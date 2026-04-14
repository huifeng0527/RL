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

# 结构 A：纯 MLP (无历史记忆，短视模型)
class MLPOnlyExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=64)
        # 只取前 16 维的标量数据，完全无视后面的历史数据
        self.net = nn.Sequential(
            nn.Linear(SCALAR_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # 手动切片：只拿标量
        scalar_part = observations[:, :SCALAR_DIM]
        return self.net(scalar_part)
    
class BiResidualGatedExtractor(BaseFeaturesExtractor):
    """
    双向残差门控特征提取器 (Bi-directional Residual Gated Fusion)
    解决多模态 RL 中的“捷径学习”与“模态坍塌”问题。
    """
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=64)
        
        self.history_channels = 2
        self.history_len = 16
        self.scalar_dim = 10  # 请核对你的真实标量维度
        self.hidden_dim = 32
        
        # 1. 空间流 (Spatial)
        self.spatial_net = nn.Sequential(
            nn.Linear(self.scalar_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim), # [修复] 统一量纲
            nn.ReLU()
        )
        
        # 2. 时序流 (Temporal)
        self.lstm_net = nn.LSTM(
            input_size=self.history_channels, 
            hidden_size=self.hidden_dim, 
            num_layers=2,
            batch_first=True
        )
        self.temporal_ln = nn.LayerNorm(self.hidden_dim) # [修复] 防止 LSTM 输出方差过大
        
        # ==========================================
        # 🌟 核心进化：双向门控生成器 (Bi-directional Gates)
        # ==========================================
        # 空间控制时间：基于当前位置，决定是否放大病理动作
        self.gate_s2t = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        
        # 时间控制空间：基于病理发作情况，决定是否增强对墙壁的敏感度
        self.gate_t2s = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid()
        )
        nn.init.constant_(self.gate_s2t[0].bias, -3.0)
        nn.init.constant_(self.gate_t2s[0].bias, -3.0)
        # 3. 融合与输出
        self.fusion_net = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, 64),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        scalar_part = observations[:, :self.scalar_dim]       
        history_part = observations[:, self.scalar_dim:]      
        
        # --- 提取基础特征 ---
        spatial_feats = self.spatial_net(scalar_part) 
        
        history_reshaped = history_part.view(-1, self.history_channels, self.history_len).permute(0, 2, 1)
        lstm_out, _ = self.lstm_net(history_reshaped)
        temporal_feats = self.temporal_ln(lstm_out[:, -1, :]) 
        
        # ==========================================
        # 🌟 核心进化：残差门控调制 (Residual Modulation)
        # ==========================================
        gate_temporal = self.gate_s2t(spatial_feats) # 取值 (0, 1)
        gate_spatial = self.gate_t2s(temporal_feats) # 取值 (0, 1)
        
        #[精髓] 使用 1.0 + gate 进行调制！
        # 即使 gate 退化为 0，特征依然能 100% 保留并传导梯度，彻底杜绝分支“脑死亡”
        modulated_temporal = temporal_feats * (1+gate_temporal) 
        modulated_spatial = spatial_feats * (1+gate_spatial)
        
        # --- 融合输出 ---
        combined = th.cat([modulated_spatial, modulated_temporal], dim=1) 
        return self.fusion_net(combined)

import torch as th
import torch.nn as nn
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GatedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=64)

        self.history_channels = 2
        self.history_len = 16
        self.scalar_dim = 10
        self.hidden_dim = 32

        # --- 1. 空间流 ---
        self.spatial_net = nn.Sequential(
            nn.Linear(self.scalar_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU()
        )

        # --- 2. 时序流 ---
        self.lstm_net = nn.LSTM(
            input_size=self.history_channels,
            hidden_size=self.hidden_dim,
            num_layers=2,
            batch_first=True
        )

        self.temporal_ln = nn.LayerNorm(self.hidden_dim)

        # --- 3. 门控 ---
        self.gate_net = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid()
        )

        # --- 4. 输出层 ---
        self.output_net = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # --- 拆分 ---
        scalar_part = observations[:, :self.scalar_dim]
        history_part = observations[:, self.scalar_dim:]

        # --- 空间特征 ---
        spatial_feats = self.spatial_net(scalar_part)  # (B, 32)

        # --- 时序特征 ---
        history_reshaped = history_part.view(
            -1, self.history_channels, self.history_len
        ).permute(0, 2, 1)  # (B, T, C)

        lstm_out, _ = self.lstm_net(history_reshaped)
        temporal_feats = self.temporal_ln(lstm_out[:, -1, :])  # (B, 32)

        # --- 门控融合 ---
        fusion_input = th.cat([spatial_feats, temporal_feats], dim=-1)

        gate = self.gate_net(fusion_input)  # (B, 32)

        fused = gate * spatial_feats + (1 - gate) * temporal_feats  # (B, 32)

        # --- 输出 ---
        output = self.output_net(fused)  # (B, 64)

        return output
        
# 结构 B：MLP + LSTM (带历史记忆)
class LSTMExtractor(BaseFeaturesExtractor):
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
        
        # 变回 (Batch, Length, Channels) 给 LSTM
        history_input = history_part.reshape(-1, HISTORY_CHANNELS, HISTORY_LEN).permute(0, 2, 1)
        
        lstm_out, _ = self.lstm_net(history_input)
        lstm_features = lstm_out[:, -1, :] # 取最后一步
        
        scalar_features = self.scalar_net(scalar_part)
        combined = th.cat([lstm_features, scalar_features], dim=1)
        
        return self.fusion_net(combined)


# 结构 C：MLP + LSTM + 辅助预测头 (全能模型)
class AuxLSTMExtractor(LSTMExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space) # 继承结构 B 的所有网络
        
        # 新增：辅助预测头 (只看 LSTM 特征，预测下一帧手部位移 dx, dy)
        self.aux_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2) 
        )

    # forward 函数继承自 LSTMExtractor，用于给 SAC 输出 64 维特征
    
    def forward_aux(self, observations: th.Tensor) -> th.Tensor:
        # ==========================================
        # 🌟 绝杀修改：复用 forward 拿到时空融合特征，直接预测！
        # ==========================================
        # 直接调用自身的 forward 方法，拿到融合了时空的 64 维特征
        fused_features = self.forward(observations)
        
        # 辅以空间信息，精准预测下一帧
        predicted_move = self.aux_head(fused_features)
        
        return predicted_move
    
class AuxGatedExtractor(GatedExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space) # 继承结构 B 的所有网络
        
        # 新增：辅助预测头 (只看 LSTM 特征，预测下一帧手部位移 dx, dy)
        self.aux_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),

        )

    # forward 函数继承自 LSTMExtractor，用于给 SAC 输出 64 维特征
    
    def forward_aux(self, observations: th.Tensor) -> th.Tensor:

        # 直接调用自身的 forward 方法，拿到融合了时空的 64 维特征
        fused_features = self.forward(observations)
        
        # 辅以空间信息，精准预测下一帧
        predicted_move = self.aux_head(fused_features)
        
        return predicted_move