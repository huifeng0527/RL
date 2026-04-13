"""Dual agent iterative training.

Alternating training between robot agent and hand agent.
Robot learns to catch hand, hand learns to avoid robot.
"""

import os
import argparse
import datetime
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.custom_env import RehabilitationEnv
from src.utils.callbacks import DebugCallback


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
    env_fns = [create_env(training_mode, robot_model, hand_model_paths) for _ in range(n_envs)]
    return DummyVecEnv(env_fns)


def train_robot(hand_model_paths, total_steps, save_path, n_envs=4):
    """Train robot agent to catch hand."""
    print(f"\n{'='*40}")
    print("Training Robot Agent")
    print(f"{'='*40}")

    vec_env = create_vec_env('robot', n_envs=n_envs, hand_model_paths=hand_model_paths)

    policy_kwargs = dict(
        net_arch=[128, 256, 64],
        share_features_extractor=True
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
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


def train_hand(robot_model_path, total_steps, save_path, n_envs=4):
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

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=3e-4,
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
    base_save_path=None
):
    """Run iterative dual agent training.

    Alternates between training robot and hand agents.
    Each iteration: train robot -> train hand -> repeat
    """
    if base_save_path is None:
        now = datetime.datetime.now().strftime("%m%d_%H%M")
        base_save_path = f"logs/dual_iterative_{now}"

    os.makedirs(base_save_path, exist_ok=True)

    robot_path = None
    hand_path = None

    for iteration in range(n_iterations):
        print(f"\n{'#'*60}")
        print(f"# Iteration {iteration + 1}/{n_iterations}")
        print(f"{'#'*60}")

        iteration_path = os.path.join(base_save_path, f"iteration_{iteration + 1}")
        os.makedirs(iteration_path, exist_ok=True)

        # Train robot with current hand opponent pool
        hand_model_paths = [hand_path] if hand_path else None
        robot_path = train_robot(
            hand_model_paths=hand_model_paths,
            total_steps=steps_per_iteration,
            save_path=os.path.join(iteration_path, "robot"),
            n_envs=n_envs
        )

        # Train hand with updated robot opponent
        hand_path = train_hand(
            robot_model_path=robot_path,
            total_steps=steps_per_iteration,
            save_path=os.path.join(iteration_path, "hand"),
            n_envs=n_envs
        )

    print(f"\n{'='*60}")
    print("Iterative Training Complete!")
    print(f"{'='*60}")
    print(f"Final robot model: {robot_path}")
    print(f"Final hand model: {hand_path}")

    return robot_path, hand_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual agent iterative training')
    parser.add_argument('--iterations', type=int, default=5, help='Number of training iterations')
    parser.add_argument('--steps', type=int, default=500000, help='Steps per iteration')
    parser.add_argument('--n_envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--save_path', type=str, default=None, help='Save path')

    args = parser.parse_args()

    run_iterative_training(
        n_iterations=args.iterations,
        steps_per_iteration=args.steps,
        n_envs=args.n_envs,
        base_save_path=args.save_path
    )
