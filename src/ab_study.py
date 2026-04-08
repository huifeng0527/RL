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
SCALAR_DIM = 16
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
    
class FiLMFeatureExtractor(BaseFeaturesExtractor):
    """
    基于 FiLM (特征级线性调制) 的双流状态提取网络。
    利用空间位置信息 (Scalars) 动态调制时序病理特征 (LSTM)。
    """
    def __init__(self, observation_space: gym.spaces.Box):
        # 最终输出维度
        super().__init__(observation_space, features_dim=64)
        
        self.history_channels = 2  # dx, dy
        self.history_len = 16      # 序列长度
        self.scalar_dim = 16       # 标量特征的维度 (请根据实际情况确认)
        
        self.lstm_hidden_dim = 32
        
        # ----------------------------------------------------
        # 1. 空间流 (Spatial Stream)
        # ----------------------------------------------------
        self.spatial_net = nn.Sequential(
            nn.Linear(self.scalar_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )
        
        # ----------------------------------------------------
        # 2. 调制发生器 (FiLM Generator) 🌟 核心创新
        # 根据空间特征，生成用于缩放(gamma)和平移(beta)的系数
        # 输出维度是 LSTM 隐藏层维度的 2 倍
        # ----------------------------------------------------
        self.film_generator = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, self.lstm_hidden_dim * 2) 
        )
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)
        
        # ----------------------------------------------------
        # 3. 时序流 (Temporal Stream - LSTM)
        # ----------------------------------------------------
        self.lstm_net = nn.LSTM(
            input_size=self.history_channels, 
            hidden_size=self.lstm_hidden_dim, 
            batch_first=True
        )
        
        # ----------------------------------------------------
        # 4. 最终融合输出 (Final Fusion)
        # ----------------------------------------------------
        # 将"空间特征"和"被调制后的时序特征"拼接，再过一层线性层
        self.fusion_net = nn.Sequential(
            nn.Linear(32 + self.lstm_hidden_dim, 64),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # --- 1. 数据切片 ---
        scalar_part = observations[:, :self.scalar_dim]       
        history_part = observations[:, self.scalar_dim:]      
        
        # --- 2. 提取空间特征 ---
        spatial_features = self.spatial_net(scalar_part) # Shape: (Batch, 32)
        
        # --- 3. 提取基础的时序特征 (LSTM) ---
        history_reshaped = history_part.view(-1, self.history_channels, self.history_len).permute(0, 2, 1)
        lstm_out, _ = self.lstm_net(history_reshaped)
        temporal_features = lstm_out[:, -1, :] # 取最后一步 (Batch, 32)
        
        # ========================================================
        # 🌟 4. 上下文特征调制 (Context-Aware Modulation via FiLM)
        # ========================================================
        # 利用空间特征生成调制参数
        film_params = self.film_generator(spatial_features) # Shape: (Batch, 64)
        
        # 切分为 gamma (缩放) 和 beta (平移)
        gamma, beta = th.chunk(film_params, 2, dim=1) # Shape: (Batch, 32), (Batch, 32)
        
        # 对时间特征进行仿射变换！
        # 这里使得网络具备了：“在墙边放大病人动作，在空旷处缩小病人动作”的能力
        modulated_temporal = (1.0 + gamma) * temporal_features + beta 
        # (注：用 1.0+gamma 是一种常见的残差技巧，使网络初始状态倾向于恒等映射)
        
        # --- 5. 融合与输出 ---
        combined = th.cat([spatial_features, modulated_temporal], dim=1) # (Batch, 64)
        return self.fusion_net(combined)

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
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2) 
        )

    # forward 函数继承自 LSTMExtractor，用于给 SAC 输出 64 维特征
    
    def forward_aux(self, observations: th.Tensor) -> th.Tensor:
        """ 专供辅助任务使用的前向传播 """
        history_part = observations[:, SCALAR_DIM:]
        history_input = history_part.view(-1, HISTORY_CHANNELS, HISTORY_LEN).permute(0, 2, 1)
        
        lstm_out, _ = self.lstm_net(history_input)
        lstm_features = lstm_out[:, -1, :]
        
        return self.aux_head(lstm_features)


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
    
class PPOAuxTrainingCallback(BaseCallback):
    def __init__(self, train_freq=1000, batch_size=512, buffer_size=10000, verbose=0):
        super().__init__(verbose)
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.optimizer = None
        
        # [核心修改 1]：PPO 没有合适的离线经验池，我们自己在内存里建一个简单的滑动窗口
        self.obs_buffer = deque(maxlen=buffer_size)
        self.next_obs_buffer = deque(maxlen=buffer_size)

    def _on_training_start(self) -> None:
        # [核心修改 2]：PPO 的特征提取器路径更简单，因为它是全局共享的
        self.extractor = self.model.policy.features_extractor
        
        # 同样为提取器单独创建一个优化器
        self.optimizer = th.optim.Adam(self.extractor.parameters(), lr=1e-5)

    def _on_step(self) -> bool:
        # ==========================================
        # 1. 收集实时数据存入自定义 Buffer
        # ==========================================
        # self.model._last_obs 是模型在执行这一步之前的观测值
        # self.locals["new_obs"] 是环境刚返回的新的观测值
        obs_array = self.model._last_obs
        next_obs_array = self.locals["new_obs"]
        
        # 因为你用了 SubprocVecEnv(n_envs=8)，这里拿到的是 [8, obs_dim] 的矩阵
        # 把并行的 8 个环境的数据拆开，一个个存入我们的 buffer
        for i in range(obs_array.shape[0]):
            self.obs_buffer.append(obs_array[i])
            self.next_obs_buffer.append(next_obs_array[i])

        # ==========================================
        # 2. 判断是否满足训练条件
        # ==========================================
        if self.num_timesteps % self.train_freq == 0 and len(self.obs_buffer) >= self.batch_size:
            
            # 从自定义 buffer 中随机抽取 batch_size 个索引
            indices = np.random.choice(len(self.obs_buffer), self.batch_size, replace=False)
            
            batch_obs = np.array([self.obs_buffer[idx] for idx in indices])
            batch_next_obs = np.array([self.next_obs_buffer[idx] for idx in indices])
            
            # 转为 PyTorch Tensor，并放入 PPO 模型所在的设备 (GPU/CPU)
            device = self.model.device
            batch_obs_tensor = th.tensor(batch_obs, dtype=th.float32).to(device)
            batch_next_obs_tensor = th.tensor(batch_next_obs, dtype=th.float32).to(device)
            
            # Ground Truth: 真正的下一帧手的位移 (取决于你的 _get_obs 拼接顺序)
            # 这里假设拼在最后两个数字
            true_next_move = batch_next_obs_tensor[:, -2:] 
            
            # 预测下一帧动作
            pred_next_move = self.extractor.forward_aux(batch_obs_tensor)
            
            # 算 Loss 并反向传播
            loss = F.l1_loss(pred_next_move, true_next_move)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # 记录 Loss
            self.logger.record("auxiliary/prediction_loss", loss.item())

        return True



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
    TOTAL_STEPS = 5_000_000
    N_ENVS = 8
    
    now = datetime.datetime.now().strftime("%m%d_%H%M")
    base_log_dir = f"logs/ablation_study_{now}/"
    os.makedirs(base_log_dir, exist_ok=True)
    
    # 定义你要跑的 3 个实验
    experiments =[
        # {"name": "1_MLP_Only", "extractor": MLPOnlyExtractor, "use_aux": False},
        {"name": "2_MLP_LSTM", "extractor": LSTMExtractor, "use_aux": False},
        # {"name": "3_MLP_LSTM_AUX", "extractor": AuxLSTMExtractor, "use_aux": True},
        {"name": "4_MLP_LSTM_FiLM", "extractor": FiLMFeatureExtractor, "use_aux": False},
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
            learning_rate=3e-4,
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
            aux_cb = PPOAuxTrainingCallback(train_freq=1000, batch_size=512)
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