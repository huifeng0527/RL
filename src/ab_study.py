import os
import datetime
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from stable_baselines3 import SAC,PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from collections import deque

# 导入你的环境
from custom_env import RehabilitationEnv

# =====================================================================
# 1. 定义三个特征提取器 (Feature Extractors)
# 假设你的 observation 是 32 维: 前 16 维是标量，后 16 维是历史轨迹 (8帧*2)
# =====================================================================
SCALAR_DIM = 10
HISTORY_LEN = 16
HISTORY_CHANNELS = 2

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
            nn.Linear(16, 2) 
        )

    # forward 函数继承自 LSTMExtractor，用于给 SAC 输出 64 维特征
    
    def forward_aux(self, observations: th.Tensor) -> th.Tensor:

        # 直接调用自身的 forward 方法，拿到融合了时空的 64 维特征
        fused_features = self.forward(observations)
        
        # 辅以空间信息，精准预测下一帧
        predicted_move = self.aux_head(fused_features)
        
        return predicted_move

# =====================================================================
# 2. 定义辅助任务的 Callback
# =====================================================================
class AuxTrainingCallback(BaseCallback):
    def __init__(self, train_freq=1000, batch_size=256, verbose=0):
        super().__init__(verbose)
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.optimizer = None

    def _on_training_start(self) -> None:
        # SAC 必须设置 share_features_extractor=True
        self.extractor = self.model.policy.actor.features_extractor
        self.optimizer = th.optim.Adam(self.extractor.parameters(), lr=5e-5)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.train_freq == 0 and self.model.replay_buffer.size() > self.batch_size:
            replay_data = self.model.replay_buffer.sample(self.batch_size)
            
            obs = replay_data.observations
            next_obs = replay_data.next_observations
            
            # Ground Truth: 真正的下一帧手的位移 (假设拼在 obs 数组的最后两个数)
            true_next_move = next_obs[:, -2:] 
            
            # 预测
            pred_next_move = self.extractor.forward_aux(obs)
            
            # 算 Loss 并反向传播
            loss = F.mse_loss(pred_next_move, true_next_move)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.logger.record("auxiliary/prediction_loss", loss.item())
        return True
    
# class PPOAuxTrainingCallback(BaseCallback):
#     def __init__(self, train_freq=1000, batch_size=512, buffer_size=10000, verbose=0):
#         super().__init__(verbose)
#         self.train_freq = train_freq
#         self.batch_size = batch_size
#         self.optimizer = None
        
#         # [核心修改 1]：PPO 没有合适的离线经验池，我们自己在内存里建一个简单的滑动窗口
#         self.obs_buffer = deque(maxlen=buffer_size)
#         self.next_obs_buffer = deque(maxlen=buffer_size)

#     def _on_training_start(self) -> None:
#         # [核心修改 2]：PPO 的特征提取器路径更简单，因为它是全局共享的
#         self.extractor = self.model.policy.features_extractor
        
#         # 同样为提取器单独创建一个优化器
#         self.optimizer = th.optim.Adam(self.extractor.parameters(), lr=1e-5)

#     def _on_step(self) -> bool:
#         # ==========================================
#         # 1. 收集实时数据存入自定义 Buffer
#         # ==========================================
#         # self.model._last_obs 是模型在执行这一步之前的观测值
#         # self.locals["new_obs"] 是环境刚返回的新的观测值
#         obs_array = self.model._last_obs
#         next_obs_array = self.locals["new_obs"]
        
#         # 因为你用了 SubprocVecEnv(n_envs=8)，这里拿到的是 [8, obs_dim] 的矩阵
#         # 把并行的 8 个环境的数据拆开，一个个存入我们的 buffer
#         for i in range(obs_array.shape[0]):
#             self.obs_buffer.append(obs_array[i])
#             self.next_obs_buffer.append(next_obs_array[i])

#         # ==========================================
#         # 2. 判断是否满足训练条件
#         # ==========================================
#         if self.num_timesteps % self.train_freq == 0 and len(self.obs_buffer) >= self.batch_size:
            
#             # 从自定义 buffer 中随机抽取 batch_size 个索引
#             indices = np.random.choice(len(self.obs_buffer), self.batch_size, replace=False)
            
#             batch_obs = np.array([self.obs_buffer[idx] for idx in indices])
#             batch_next_obs = np.array([self.next_obs_buffer[idx] for idx in indices])
            
#             # 转为 PyTorch Tensor，并放入 PPO 模型所在的设备 (GPU/CPU)
#             device = self.model.device
#             batch_obs_tensor = th.tensor(batch_obs, dtype=th.float32).to(device)
#             batch_next_obs_tensor = th.tensor(batch_next_obs, dtype=th.float32).to(device)
            
#             # Ground Truth: 真正的下一帧手的位移 (取决于你的 _get_obs 拼接顺序)
#             # 这里假设拼在最后两个数字
#             true_next_move = batch_next_obs_tensor[:, -2:] 
            
#             # 预测下一帧动作
#             pred_next_move = self.extractor.forward_aux(batch_obs_tensor)
            
#             # 算 Loss 并反向传播
#             loss = F.l1_loss(pred_next_move, true_next_move)
#             self.optimizer.zero_grad()
#             loss.backward()
#             self.optimizer.step()
            
#             # 记录 Loss
#             self.logger.record("auxiliary/prediction_loss", loss.item())

#         return True

class PPOAuxTrainingCallback(BaseCallback):
    def __init__(self, batch_size=256, buffer_size=10000, aux_epochs=3, verbose=0):
        super().__init__(verbose)
        self.batch_size = batch_size
        self.aux_epochs = aux_epochs # 每次 rollout 结束，辅助任务复习几次
        self.optimizer = None
        
        self.obs_buffer = deque(maxlen=buffer_size)
        self.next_obs_buffer = deque(maxlen=buffer_size)

    def _on_training_start(self) -> None:
        self.extractor = self.model.policy.features_extractor
        # 保持较低的学习率，防止辅助梯度淹没 RL 梯度
        self.optimizer = th.optim.Adam(self.extractor.parameters(), lr=5e-5)

    def _on_step(self) -> bool:
        # 在 _on_step 中 【只收集数据，绝对不训练网络！】
        obs_array = self.model._last_obs
        next_obs_array = self.locals["new_obs"]
        
        for i in range(obs_array.shape[0]):
            self.obs_buffer.append(obs_array[i])
            self.next_obs_buffer.append(next_obs_array[i])
        return True

    def _on_rollout_end(self) -> None:
        # =========================================================
        # 🌟 核心修复：只在 PPO 收集完数据、准备更新网络的时刻，同步进行辅助任务更新
        # 这保证了 PPO 在收集数据的 512 步期间，底层特征提取器是绝对冻结且稳定的！
        # =========================================================
        if len(self.obs_buffer) >= self.batch_size:
            epoch_losses =[]
            
            # 在这里做几个 epoch 的辅助任务更新
            for _ in range(self.aux_epochs):
                indices = np.random.choice(len(self.obs_buffer), self.batch_size, replace=False)
                batch_obs = np.array([self.obs_buffer[idx] for idx in indices])
                batch_next_obs = np.array([self.next_obs_buffer[idx] for idx in indices])
                
                device = self.model.device
                batch_obs_tensor = th.tensor(batch_obs, dtype=th.float32).to(device)
                batch_next_obs_tensor = th.tensor(batch_next_obs, dtype=th.float32).to(device)
                
                # Ground Truth
                true_next_move = batch_next_obs_tensor[:, -2:] 
                
                # 预测与 Loss (乘以 100 放大数值)
                pred_next_move = self.extractor.forward_aux(batch_obs_tensor)
                loss = F.mse_loss(pred_next_move, true_next_move)
                
                self.optimizer.zero_grad()
                loss.backward()
                # 裁剪梯度，防止破坏 PPO 刚更新完的特征
                th.nn.utils.clip_grad_norm_(self.extractor.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                epoch_losses.append(loss.item())
            
            # 记录平均 Loss 到 Tensorboard
            self.logger.record("auxiliary/prediction_loss", np.mean(epoch_losses))

class SaveMetricsCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.data = {
            "timesteps": [],
            "rewards": [],
            "ep_lengths": []
        }

    def _on_step(self) -> bool:
        # 从 infos 里拿 episode 信息（Monitor 提供）
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
        
        print(f"✅ Metrics saved to {self.save_path}/metrics.npz")

# =====================================================================
# 3. 自动化消融实验主控脚本
# =====================================================================
def run_ablation_study():
    # 实验配置
    TOTAL_STEPS = 8_000_000
    N_ENVS = 8
    
    now = datetime.datetime.now().strftime("%m%d_%H%M")
    base_log_dir = f"logs/ablation_study_{now}/"
    os.makedirs(base_log_dir, exist_ok=True)
    
    # 定义你要跑的 3 个实验
    experiments =[
        {"name": "1_MLP_Only", "extractor": MLPOnlyExtractor, "use_aux": False},
        {"name": "2_MLP_LSTM", "extractor": LSTMExtractor, "use_aux": False},
        {"name": "3_MLP_LSTM_AUX", "extractor": AuxLSTMExtractor, "use_aux": True},
        {"name": "4_MLP_LSTM_gate", "extractor": GatedExtractor, "use_aux": False},
        {"name": "5_MLP_LSTM_gate_AUX", "extractor": AuxGatedExtractor, "use_aux": True},
    ]
    
    for exp in experiments:
        exp_name = exp["name"]
        print(f"\n{'='*40}")
        print(f"🚀 开始消融实验: {exp_name}")
        print(f"{'='*40}\n")
        
        # 1. 创建环境 (这里统一对抗随机脚本或某一个固定的Hand，保证变量唯一)
        def make_env():
            env = RehabilitationEnv(training_mode='robot')
            # 设定固定的测试对手参数，保证实验公平
            # env.patient.params = {'v_max': 1.0, 'tremor_amp': 0.5, 'delay_steps': 2}
            # env.patient_locked = True 
            return Monitor(env)
            
        vec_env = SubprocVecEnv([make_env for _ in range(N_ENVS)])
        
        # 2. 网络参数配置 (强制共享提取器)
        policy_kwargs = dict(
            net_arch=[128, 256, 64],
            features_extractor_class=exp["extractor"],
            features_extractor_kwargs=dict(),
            share_features_extractor=True # ⚠️ 极其关键！
        )
        
        # 3. 路径配置
        save_path = os.path.join(base_log_dir, exp_name)
        tb_dir = os.path.join(base_log_dir, "tensorboard_logs")
        os.makedirs(save_path, exist_ok=True)
        
        # 4. 初始化模型
        model = PPO(
            "MlpPolicy", 
            vec_env, 
            verbose=0, 
            learning_rate=1e-4,
            batch_size=512,
            # ent_coef='auto', 
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_dir
        )
        
        # 5. 配置 Callbacks
        eval_cb = EvalCallback(
            vec_env, best_model_save_path=save_path,
            log_path=save_path, eval_freq=10000,n_eval_episodes=100,
            deterministic=True, render=False
        )
        metrics_cb = SaveMetricsCallback(save_path)
        if exp["use_aux"]:
            aux_cb = PPOAuxTrainingCallback(batch_size=512)
            callbacks = CallbackList([eval_cb, aux_cb,metrics_cb])
        else:
            callbacks = CallbackList([eval_cb, metrics_cb])
            
        # 6. 开启训练
        # tb_log_name 会让三根曲线出现在同一个 Tensorboard 图表中！
        model.learn(total_timesteps=TOTAL_STEPS, callback=callbacks, tb_log_name=exp_name)
        
        # 7. 清理内存，准备跑下一个实验
        model.save(os.path.join(save_path, "final_model.zip"))
        vec_env.close()
        del model
        
    print("\n🎉 所有消融实验运行完毕！请使用 Tensorboard 查看对比曲线。")
    print(f"tensorboard --logdir {tb_dir}")

if __name__ == '__main__':
    run_ablation_study()