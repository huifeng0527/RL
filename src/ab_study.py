"""Ablation study framework for rehabilitation RL.

Runs experiments with different feature extractor architectures against
a fixed scripted hand with biomechanical filtering (CMD-DR motor layer).

Metrics: Reward, Episode Length, ZPD Coverage, Workspace Coverage
"""

import os
import datetime
import json
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.custom_env import RehabilitationEnv
from src.utils.feature_extractors import MLPOnlyExtractor, LSTMExtractor, AuxLSTMExtractor
from src.utils.ablation_callbacks import PPOAuxTrainingCallback, SaveMetricsCallback


class AblationMetricsCallback(BaseCallback):
    """Track ZPD coverage and workspace coverage during evaluation."""

    def __init__(self, save_path, eval_freq=10000, n_eval_episodes=50, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.zpd_min = 4.0
        self.zpd_max = 6.0
        self.results = []
        self.best = {
            "reward": None,
            "zpd_coverage": None,
            "workspace_coverage": None,
        }

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        # Create a single evaluation environment
        eval_env = RehabilitationEnv(
            training_mode='robot',
            pathology_mode='impaired'
        )
        eval_env.hand_move_epsilon = 0.15
        eval_env.random_noise = True
        eval_env.noise_sigma = 0.03
        eval_env.stride_hand_random = [0.12, 0.28]
        eval_env.stride_robot_random = [0.2, 0.5]

        # Run evaluation episodes
        zpd_coverages = []
        workspace_coverages = []
        episode_lengths = []
        rewards = []

        for _ in range(self.n_eval_episodes):
            obs, info = eval_env.reset()
            done = False
            total_reward = 0
            steps = 0
            in_zpd_count = 0

            visited_x = []
            visited_y = []

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1

                dist = info.get('dist', 0)
                if self.zpd_min <= dist <= self.zpd_max:
                    in_zpd_count += 1

                robot_pos = info.get('robot_pos', np.zeros(2))
                visited_x.append(robot_pos[0])
                visited_y.append(robot_pos[1])

            zpd_coverage = in_zpd_count / steps if steps > 0 else 0
            zpd_coverages.append(zpd_coverage)
            episode_lengths.append(steps)
            rewards.append(total_reward)

            # Workspace coverage (bounding box area)
            if len(visited_x) > 1:
                x_range = max(visited_x) - min(visited_x)
                y_range = max(visited_y) - min(visited_y)
                # Normalize by max possible area (env is 15x10 with margin 0.3)
                max_area = (15 - 0.6) * (10 - 0.6)
                workspace_cov = (x_range * y_range) / max_area
                workspace_coverages.append(workspace_cov)
            else:
                workspace_coverages.append(0)

        # Log metrics
        result = {
            'timestep': self.num_timesteps,
            'reward_mean': float(np.mean(rewards)),
            'reward_std': float(np.std(rewards)),
            'episode_length_mean': float(np.mean(episode_lengths)),
            'episode_length_std': float(np.std(episode_lengths)),
            'zpd_coverage_mean': float(np.mean(zpd_coverages)),
            'zpd_coverage_std': float(np.std(zpd_coverages)),
            'workspace_coverage_mean': float(np.mean(workspace_coverages)),
            'workspace_coverage_std': float(np.std(workspace_coverages)),
        }
        self.results.append(result)
        self._update_best(result)
        self.save_results(self.save_path)

        self.logger.record("eval/reward_mean", result['reward_mean'])
        self.logger.record("eval/episode_length_mean", result['episode_length_mean'])
        self.logger.record("eval/zpd_coverage_mean", result['zpd_coverage_mean'])
        self.logger.record("eval/workspace_coverage_mean", result['workspace_coverage_mean'])

        if self.verbose > 0:
            print(f"\n[Eval {self.num_timesteps}] Reward: {result['reward_mean']:.2f} | "
                  f"Length: {result['episode_length_mean']:.1f} | "
                  f"ZPD: {result['zpd_coverage_mean']:.1%} | "
                  f"Workspace: {result['workspace_coverage_mean']:.2%}")

        return True

    def _update_best(self, result):
        if self.best["reward"] is None or result["reward_mean"] > self.best["reward"]["reward_mean"]:
            self.best["reward"] = result
        if self.best["zpd_coverage"] is None or result["zpd_coverage_mean"] > self.best["zpd_coverage"]["zpd_coverage_mean"]:
            self.best["zpd_coverage"] = result
        if self.best["workspace_coverage"] is None or result["workspace_coverage_mean"] > self.best["workspace_coverage"]["workspace_coverage_mean"]:
            self.best["workspace_coverage"] = result

    def save_results(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'ablation_metrics.json'), 'w') as f:
            json.dump(self.results, f, indent=2)
        with open(os.path.join(path, 'ablation_best_metrics.json'), 'w') as f:
            json.dump(self.best, f, indent=2)


def run_ablation_study():
    # Experiment configuration
    TOTAL_STEPS = 8_000_000
    N_ENVS = 8
    EVAL_FREQ = 50_000
    N_EVAL_EPISODES = 50

    now = datetime.datetime.now().strftime("%m%d_%H%M")
    base_log_dir = f"logs/ablation_study_{now}/"
    os.makedirs(base_log_dir, exist_ok=True)

    experiments = [
        {"name": "1_MLP_Only", "extractor": MLPOnlyExtractor, "use_aux": False},
        {"name": "2_MLP_LSTM", "extractor": LSTMExtractor, "use_aux": False},
        {"name": "3_MLP_LSTM_AUX", "extractor": AuxLSTMExtractor, "use_aux": True},
    ]

    for exp in experiments:
        exp_name = exp["name"]
        print(f"\n{'='*60}")
        print(f"Starting ablation experiment: {exp_name}")
        print(f"{'='*60}\n")

        # Create environment with fixed scripted hand + CMD-DR motor layer
        def make_env():
            env = RehabilitationEnv(
                training_mode='robot',
                pathology_mode='impaired'
            )
            # Fixed hand parameters for fair comparison
            # env.hand_move_epsilon = 0.15
            # env.random_noise = True
            # env.noise_sigma = 0.03
            # env.stride_hand_random = [0.12, 0.28]
            # env.stride_robot_random = [0.2, 0.5]
            return Monitor(env)

        vec_env = SubprocVecEnv([make_env for _ in range(N_ENVS)])

        # Network configuration
        policy_kwargs = dict(
            net_arch=[256, 256, 256, 64],
            features_extractor_class=exp["extractor"],
            features_extractor_kwargs=dict(),
            share_features_extractor=True
        )

        # Paths
        save_path = os.path.join(base_log_dir, exp_name)
        tb_dir = os.path.join(base_log_dir, "tensorboard_logs")
        os.makedirs(save_path, exist_ok=True)

        # Initialize model
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=1e-4,
            batch_size=512,
            n_steps=2048,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_dir
        )

        # Configure callbacks
        ablation_cb = AblationMetricsCallback(
            save_path=save_path,
            eval_freq=EVAL_FREQ,
            n_eval_episodes=N_EVAL_EPISODES,
            verbose=1
        )

        callbacks = [ablation_cb]

        if exp["use_aux"]:
            aux_cb = PPOAuxTrainingCallback(batch_size=512, aux_epochs=3)
            callbacks.append(aux_cb)

        # Train
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=CallbackList(callbacks),
            tb_log_name=exp_name
        )

        # Save results
        model.save(os.path.join(save_path, "final_model.zip"))
        model.save(os.path.join(save_path, "best_model.zip"))
        ablation_cb.save_results(save_path)

        # Cleanup
        vec_env.close()
        del model

    print("\n" + "="*60)
    print("All ablation experiments completed!")
    print(f"Results saved to: {base_log_dir}")
    print(f"View training curves: tensorboard --logdir {tb_dir}")
    print("="*60)



if __name__ == "__main__":
    run_ablation_study()
