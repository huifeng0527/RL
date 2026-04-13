"""Training utilities for rehabilitation environment."""

import os
import json
from collections import deque
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from .custom_env import RehabilitationEnv
from .utils.callbacks import DebugCallback


def make_env(training_mode='robot', robot_model=None, hand_model_paths=None):
    """Factory function to create a single environment instance."""
    def _init():
        env = RehabilitationEnv(
            training_mode=training_mode,
            robot_model=robot_model,
            hand_model_paths=hand_model_paths
        )
        return env
    return _init


def create_vec_env(training_mode='robot', n_envs=1, robot_model=None, hand_model_paths=None):
    """Create a vectorized environment with monitoring."""
    env_fns = [make_env(training_mode, robot_model, hand_model_paths) for _ in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    return vec_env


def setup_training(
    training_mode='robot',
    algorithm='PPO',
    n_envs=8,
    hand_model_paths=None,
    save_path='logs/best_model',
    eval_freq=10000,
    total_timesteps=5000000,
    hand_model=None
):
    """Setup and run training for either robot or hand agent.

    Args:
        training_mode: 'robot' or 'hand'
        algorithm: 'PPO' or 'SAC'
        n_envs: Number of parallel environments
        hand_model_paths: List of paths to opponent hand models (for robot training)
        save_path: Base path for saving models
        eval_freq: Evaluation frequency in steps
        total_timesteps: Total training steps
        hand_model: Pre-loaded hand model (for hand training with robot opponent)
    """
    os.makedirs(save_path, exist_ok=True)

    if training_mode == 'robot':
        robot_training_env = create_vec_env('robot', n_envs=n_envs, hand_model_paths=hand_model_paths)
        eval_env = create_vec_env('robot', n_envs=1)

        if algorithm == 'PPO':
            model = PPO(
                "MlpPolicy",
                robot_training_env,
                learning_rate=3e-4,
                batch_size=512,
                n_steps=2048,
                verbose=1
            )
        else:
            model = SAC("MlpPolicy", robot_training_env, verbose=1)

    elif training_mode == 'hand':
        if hand_model is None:
            raise ValueError("hand_model is required for hand training")

        hand_training_env = create_vec_env('hand', n_envs=n_envs, robot_model=hand_model)
        eval_env = create_vec_env('hand', n_envs=1, robot_model=hand_model)

        if algorithm == 'PPO':
            model = PPO(
                "MlpPolicy",
                hand_training_env,
                learning_rate=3e-4,
                batch_size=512,
                n_steps=2048,
                verbose=1
            )
        else:
            model = SAC("MlpPolicy", hand_training_env, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )

    debug_callback = DebugCallback(env=eval_env, log_freq=eval_freq, verbose=1)
    callback = CallbackList([eval_callback, debug_callback])

    return model, callback, save_path


def train_robot(
    hand_model_paths=None,
    algorithm='PPO',
    n_envs=8,
    total_timesteps=5000000,
    save_path='logs/robot_model',
    eval_freq=10000
):
    """Train robot agent to catch the hand."""
    model, callback, save_path = setup_training(
        training_mode='robot',
        algorithm=algorithm,
        n_envs=n_envs,
        hand_model_paths=hand_model_paths,
        save_path=save_path,
        eval_freq=eval_freq,
        total_timesteps=total_timesteps
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)

    settings = {
        'training_mode': 'robot',
        'algorithm': algorithm,
        'n_envs': n_envs,
        'hand_model_paths': hand_model_paths,
        'total_timesteps': total_timesteps
    }

    with open(os.path.join(save_path, "settings.json"), "w") as f:
        json.dump(settings, f, indent=4)

    return model


def train_hand(
    robot_model_path,
    algorithm='PPO',
    n_envs=8,
    total_timesteps=5000000,
    save_path='logs/hand_model',
    eval_freq=10000
):
    """Train hand agent to avoid the robot."""
    robot_model = PPO.load(robot_model_path, custom_objects={'learning_rate': 0.0, 'optimizer_class': None})

    model, callback, save_path = setup_training(
        training_mode='hand',
        algorithm=algorithm,
        n_envs=n_envs,
        hand_model=robot_model,
        save_path=save_path,
        eval_freq=eval_freq,
        total_timesteps=total_timesteps
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)

    settings = {
        'training_mode': 'hand',
        'algorithm': algorithm,
        'robot_model_path': robot_model_path,
        'n_envs': n_envs,
        'total_timesteps': total_timesteps
    }

    with open(os.path.join(save_path, "settings.json"), "w") as f:
        json.dump(settings, f, indent=4)

    return model
