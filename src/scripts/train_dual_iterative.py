"""Dual agent iterative training.

Alternating training between robot agent and hand agent.
Robot learns to catch hand, hand learns to avoid robot.
"""

import sys
import os

# Add project root to path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from src.custom_env import RehabilitationEnv
from src.utils.callbacks import DebugCallback
from src.utils.feature_extractors import MLPOnlyExtractor, LSTMExtractor, AuxLSTMExtractor, GatedExtractor, AuxGatedExtractor

EXTRACTOR_MAP = {
    "mlp": MLPOnlyExtractor,
    "lstm": LSTMExtractor,
    "aux_lstm": AuxLSTMExtractor,
    "gate": GatedExtractor,
    "aux_gate": AuxGatedExtractor,
}


def create_env(training_mode, robot_model=None, hand_model_paths=None):
    """Create a single environment instance."""
    def _make():
        env = RehabilitationEnv(
            training_mode=training_mode,
            robot_model=robot_model,
            hand_model_paths=hand_model_paths
        )
        return Monitor(env)
    return _make


def create_vec_env(training_mode, n_envs=4, robot_model=None, hand_model_paths=None):
    """Create vectorized environment."""
    env_fns =[create_env(training_mode, robot_model, hand_model_paths) for _ in range(n_envs)]
    return DummyVecEnv(env_fns)


def train_robot(hand_model_paths, total_steps, save_path, n_envs=4, extractor_class=None, previous_robot_path=None):
    """Train robot agent to catch hand."""
    print(f"\n{'='*40}")
    print("Training Robot Agent")
    print(f"{'='*40}")

    # 传入整个对手池
    vec_env = create_vec_env('robot', n_envs=n_envs, hand_model_paths=hand_model_paths)

    policy_kwargs = dict(
        net_arch=[128, 256, 64],
        share_features_extractor=True
    )
    if extractor_class is not None:
        policy_kwargs["features_extractor_class"] = extractor_class

    # ========================================================
    # [核心修改 1]：Robot 必须继承上一轮的大脑，持续进化！
    # ========================================================
    if previous_robot_path and os.path.exists(previous_robot_path):
        print(f"[*] 继承记忆: 加载上一代 Robot 模型 -> {previous_robot_path}")
        model = PPO.load(previous_robot_path, env=vec_env, custom_objects={'learning_rate': 3e-4, 'ent_coef': 0.01})
        # 恢复 Logger 防止 Tensorboard 报错/断裂
        logger = configure(os.path.join(save_path, "tensorboard"), ["stdout", "tensorboard"])
        model.set_logger(logger)
    else:
        print("[*] 初始纪元: 创建全新的 Robot 模型")
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            learning_rate=3e-4,
            ent_coef=0.01,
            batch_size=512,
            n_steps=2048,
            policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(save_path, "tensorboard")
        )

    eval_env = create_vec_env('robot', n_envs=1, hand_model_paths=hand_model_paths)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, "robot"),
        eval_freq=10000,
        deterministic=True,
        n_eval_episodes=10
    )
    debug_callback = DebugCallback(env=eval_env, log_freq=10000)

    model.learn(
        total_timesteps=total_steps,
        callback=CallbackList([eval_callback, debug_callback])
    )

    robot_path = os.path.join(save_path, "robot", "best_model.zip")
    model.save(robot_path)
    print(f"Robot model saved to {robot_path}")

    vec_env.close()
    eval_env.close()

    return robot_path


def train_hand(robot_model_path, total_steps, save_path, n_envs=4, extractor_class=None):
    """Train hand agent to avoid robot."""
    print(f"\n{'='*40}")
    print("Training Hand Agent")
    print(f"{'='*40}")

    # Load robot model as opponent
    robot_model = PPO.load(
        robot_model_path,
        custom_objects={'learning_rate': 0.0, 'optimizer_class': None}
    )

    vec_env = create_vec_env('hand', n_envs=n_envs, robot_model=robot_model)

    policy_kwargs = dict(
        net_arch=[128, 256, 64],
        share_features_extractor=True
    )
    if extractor_class is not None:
        policy_kwargs["features_extractor_class"] = extractor_class

    # ========================================================
    # [核心机制 2]：Hand 每次必须从头开始训练 (Train from scratch)
    # 不加载历史 Hand 模型，逼迫它针对当前 Robot 寻找全新破绽！
    # ========================================================
    print("[*] 基因变异: 创建全新的 Hand 对手模型 (从头开始)")
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=3e-4,
        # ent_coef=0.01,
        batch_size=512,
        n_steps=2048,
        policy_kwargs=policy_kwargs,
        tensorboard_log=os.path.join(save_path, "tensorboard")
    )

    eval_env = create_vec_env('hand', n_envs=1, robot_model=robot_model)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, "hand"),
        eval_freq=10000,
        deterministic=True,
        n_eval_episodes=10
    )
    debug_callback = DebugCallback(env=eval_env, log_freq=10000)

    model.learn(
        total_timesteps=total_steps,
        callback=CallbackList([eval_callback, debug_callback])
    )

    hand_path = os.path.join(save_path, "hand", "best_model.zip")
    model.save(hand_path)
    print(f"Hand model saved to {hand_path}")

    vec_env.close()
    eval_env.close()

    return hand_path


def run_iterative_training(
    n_iterations=5,
    steps_per_iteration=500000,
    n_envs=4,
    base_save_path=None,
    extractor_name="mlp"
):
    """Run iterative dual agent training.

    Alternates between training robot and hand agents.
    Each iteration: train robot -> train hand -> repeat
    """
    if base_save_path is None:
        now = datetime.datetime.now().strftime("%m%d_%H%M")
        base_save_path = f"logs/dual_iterative_{now}"

    os.makedirs(base_save_path, exist_ok=True)

    # Get extractor class
    extractor_class = EXTRACTOR_MAP.get(extractor_name.lower())
    if extractor_class is None:
        print(f"Warning: Unknown extractor '{extractor_name}', using default (no custom extractor)")
        extractor_class = None
    else:
        print(f"Using feature extractor: {extractor_class.__name__}")

    # ========================================================
    # [核心修改 3]：维护对手池 (Opponent Pool) 和 Robot 血脉
    # ========================================================
    historical_hand_paths =[] # 历史对手池
    current_robot_path = None  # 用于向下一轮传递的 Robot 记忆

    for iteration in range(n_iterations):
        print(f"\n{'#'*60}")
        print(f"# Iteration {iteration + 1}/{n_iterations}")
        print(f"{'#'*60}")

        iteration_path = os.path.join(base_save_path, f"iteration_{iteration + 1}")
        os.makedirs(iteration_path, exist_ok=True)
        
        print(f"➤ 当前对手池大小: {len(historical_hand_paths) + 1} (包含底层参数化脚本手)")

        # 1. 训练 Robot (传入对手池 & 上一代自己的脑子)
        current_robot_path = train_robot(
            hand_model_paths=historical_hand_paths,
            total_steps=steps_per_iteration,
            save_path=os.path.join(iteration_path, "robot"),
            n_envs=n_envs,
            extractor_class=extractor_class,
            previous_robot_path=current_robot_path # <--- 让 Robot 越来越强
        )

        # 2. 训练 Hand (只针对最新代 Robot 找破绽)
        new_hand_path = train_hand(
            robot_model_path=current_robot_path,
            total_steps=steps_per_iteration,
            save_path=os.path.join(iteration_path, "hand"),
            n_envs=n_envs,
            extractor_class=extractor_class
        )

        # 3. 将新炼成的“克星 Hand”加入对手池
        if new_hand_path not in historical_hand_paths:
            historical_hand_paths.append(new_hand_path)
            print(f"[*] 注入新鲜血液: 已将新 Hand 添加入联盟对手池！")

    print(f"\n{'='*60}")
    print("Iterative League Training Complete!")
    print(f"{'='*60}")
    print(f"Final Robot Master Model: {current_robot_path}")
    print(f"Total Opponents Generated: {len(historical_hand_paths)}")

    return current_robot_path, historical_hand_paths[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual agent iterative league training')
    parser.add_argument('--iterations', type=int, default=5, help='Number of training iterations')
    parser.add_argument('--steps', type=int, default=5_000_000, help='Steps per iteration')
    parser.add_argument('--n_envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--save_path', type=str, default=None, help='Save path')
    parser.add_argument('--extractor', type=str, default='lstm',
                        choices=['mlp', 'lstm', 'aux_lstm', 'gate', 'aux_gate'],
                        help='Feature extractor to use')

    args = parser.parse_args()

    run_iterative_training(
        n_iterations=args.iterations,
        steps_per_iteration=args.steps,
        n_envs=args.n_envs,
        base_save_path=args.save_path,
        extractor_name=args.extractor
    )