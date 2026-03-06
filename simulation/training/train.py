"""
训练脚本
用于训练强化学习模型
"""
import os
import yaml
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor

from simulation.environments import CustomEnv
from simulation.utils.config_loader import SimulationConfigLoader


def create_env(config):
    """创建环境"""
    env = CustomEnv(config=config)
    return env


def train_model(config_dir=None):
    """
    训练模型
    
    Args:
        config_dir: 配置文件目录
    """
    # 加载配置
    config_loader = SimulationConfigLoader(config_dir)
    env_config = config_loader.get_env_config()
    train_config = config_loader.get_train_config()
    
    # 创建环境
    env = create_env(env_config)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    env = VecMonitor(env)
    
    # 创建模型
    algorithm = train_config.get('algorithm', 'SAC')
    policy = train_config.get('policy', 'MlpPolicy')
    
    if algorithm == 'SAC':
        model = SAC(
            policy,
            env,
            learning_rate=train_config.get('learning_rate', 0.0003),
            buffer_size=train_config.get('buffer_size', 1000000),
            learning_starts=train_config.get('learning_starts', 100),
            batch_size=train_config.get('batch_size', 256),
            tau=train_config.get('tau', 0.005),
            gamma=train_config.get('gamma', 0.99),
            ent_coef=train_config.get('ent_coef', 'auto'),
            target_update_interval=train_config.get('target_update_interval', 1),
            train_freq=train_config.get('train_freq', 1),
            gradient_steps=train_config.get('gradient_steps', 1),
            verbose=train_config.get('verbose', 1),
            tensorboard_log=train_config.get('tensorboard_log', None)
        )
    elif algorithm == 'PPO':
        model = PPO(
            policy,
            env,
            learning_rate=train_config.get('learning_rate', 0.0003),
            n_steps=train_config.get('n_steps', 2048),
            batch_size=train_config.get('batch_size', 64),
            n_epochs=train_config.get('n_epochs', 10),
            gamma=train_config.get('gamma', 0.99),
            verbose=train_config.get('verbose', 1),
            tensorboard_log=train_config.get('tensorboard_log', None)
        )
    else:
        raise ValueError(f"不支持的算法: {algorithm}")
    
    # 创建评估回调
    eval_env = create_env(env_config)
    eval_env = Monitor(eval_env)
    eval_env = DummyVecEnv([lambda: eval_env])
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=train_config.get('best_model_save_path', None),
        log_path=train_config.get('eval_log_path', None),
        eval_freq=train_config.get('eval_freq', 10000),
        n_eval_episodes=train_config.get('n_eval_episodes', 10),
        deterministic=True,
        render=False
    )
    
    # 训练
    total_timesteps = train_config.get('total_timesteps', 100000)
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        log_interval=10
    )
    
    # 保存模型
    save_path = train_config.get('save_path', './logs')
    os.makedirs(save_path, exist_ok=True)
    model.save(os.path.join(save_path, 'final_model'))
    
    print(f"训练完成！模型已保存到: {save_path}")
    
    # 保存环境参数
    env.unwrapped.vec_envs[0].env.env.save_args(save_path)


if __name__ == "__main__":
    train_model()

