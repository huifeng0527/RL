"""Ablation study framework for rehabilitation RL.

Runs experiments with different feature extractor architectures.
"""

import os
import datetime
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from .custom_env import RehabilitationEnv
from .utils.feature_extractors import MLPOnlyExtractor, LSTMExtractor, AuxLSTMExtractor, GatedExtractor, AuxGatedExtractor
from .utils.ablation_callbacks import PPOAuxTrainingCallback, SaveMetricsCallback


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



if __name__ == "__main__":
    run_ablation_study()
