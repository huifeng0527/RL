"""Ablation study framework for rehabilitation RL."""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.custom_env import RehabilitationEnv
from src.observation_schema import HISTORY_CHANNELS, INTERACTION_HISTORY_CHANNELS
from src.utils.ablation_callbacks import PPOFutureAuxTrainingCallback
from src.utils.feature_extractors import MLPOnlyExtractor, StrategyGRUAuxExtractor


DEFAULT_HAND_PATH = (
    "logs/league_paper_gru_multistep_aux_pfsp_window_20iter/"
    "iteration_10/hand/hand/best_model.zip"
)


class AblationMetricsCallback(BaseCallback):
    """Track evaluation metrics during ablation training."""

    def __init__(
        self,
        save_path,
        hand_model_paths=None,
        history_length=16,
        history_mode="motion",
        eval_freq=10000,
        n_eval_episodes=50,
        scripted_hand_sample_prob=0.0,
        verbose=0,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.hand_model_paths = hand_model_paths
        self.history_length = int(history_length)
        self.history_mode = str(history_mode)
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.scripted_hand_sample_prob = scripted_hand_sample_prob
        self.zpd_min = None
        self.zpd_max = None
        self.results = []
        self.best = {
            "reward": None,
            "zpd_coverage": None,
            "workspace_coverage": None,
        }

    def _make_eval_env(self):
        env = RehabilitationEnv(
            training_mode="robot",
            hand_model_paths=self.hand_model_paths,
            history_length=self.history_length,
            history_mode=self.history_mode,
        )
        env.scripted_hand_sample_prob = self.scripted_hand_sample_prob
        return env

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        eval_env = self._make_eval_env()
        zpd_min = float(eval_env.zpd_min)
        zpd_max = float(eval_env.zpd_max)
        zpd_coverages = []
        workspace_coverages = []
        episode_lengths = []
        rewards = []

        for _ in range(self.n_eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            total_reward = 0.0
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

                dist = info.get("dist", 0.0)
                if zpd_min <= dist <= zpd_max:
                    in_zpd_count += 1

                robot_pos = info.get("robot_pos", np.zeros(2))
                visited_x.append(robot_pos[0])
                visited_y.append(robot_pos[1])

            zpd_coverages.append(in_zpd_count / steps if steps > 0 else 0.0)
            episode_lengths.append(steps)
            rewards.append(total_reward)

            if len(visited_x) > 1:
                x_range = max(visited_x) - min(visited_x)
                y_range = max(visited_y) - min(visited_y)
                max_area = (15 - 0.6) * (10 - 0.6)
                workspace_coverages.append((x_range * y_range) / max_area)
            else:
                workspace_coverages.append(0.0)

        eval_env.close()
        result = {
            "timestep": self.num_timesteps,
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "episode_length_mean": float(np.mean(episode_lengths)),
            "episode_length_std": float(np.std(episode_lengths)),
            "zpd_coverage_mean": float(np.mean(zpd_coverages)),
            "zpd_coverage_std": float(np.std(zpd_coverages)),
            "workspace_coverage_mean": float(np.mean(workspace_coverages)),
            "workspace_coverage_std": float(np.std(workspace_coverages)),
        }
        self.results.append(result)
        self._update_best(result)
        self.save_results(self.save_path)

        self.logger.record("eval/reward_mean", result["reward_mean"])
        self.logger.record("eval/episode_length_mean", result["episode_length_mean"])
        self.logger.record("eval/zpd_coverage_mean", result["zpd_coverage_mean"])
        self.logger.record("eval/workspace_coverage_mean", result["workspace_coverage_mean"])

        if self.verbose > 0:
            print(
                f"\n[Eval {self.num_timesteps}] Reward: {result['reward_mean']:.2f} | "
                f"Length: {result['episode_length_mean']:.1f} | "
                f"ZPD: {result['zpd_coverage_mean']:.1%} | "
                f"Workspace: {result['workspace_coverage_mean']:.2%}"
            )

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
        with open(os.path.join(path, "ablation_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)
        with open(os.path.join(path, "ablation_best_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(self.best, f, indent=2)


def make_robot_env(hand_model_paths, history_length, history_mode, scripted_hand_sample_prob):
    def _init():
        env = RehabilitationEnv(
            training_mode="robot",
            hand_model_paths=hand_model_paths,
            history_length=history_length,
            history_mode=history_mode,
        )
        env.scripted_hand_sample_prob = scripted_hand_sample_prob
        return Monitor(env)

    return _init


def run_ablation_study(args):
    hand_model_paths = [args.hand_path] if args.hand_path else None
    if args.hand_path and not os.path.exists(args.hand_path):
        raise FileNotFoundError(f"Hand model not found: {args.hand_path}")

    now = datetime.datetime.now().strftime("%m%d_%H%M")
    run_name = args.run_name or ("ablation_h10_hand" if args.hand_path else "ablation_study")
    base_log_dir = args.output_dir or f"logs/{run_name}_{now}"
    os.makedirs(base_log_dir, exist_ok=True)

    experiments = [
        {"name": "1_MLP_Only", "extractor": MLPOnlyExtractor, "use_aux": False},
        # {"name": "2_MLP_LSTM", "extractor": LSTMExtractor, "use_aux": False},
        # {"name": "3_MLP_LSTM_AUX", "extractor": AuxLSTMExtractor, "use_aux": True},
        {"name": "4_StrategyGRU", "extractor": StrategyGRUAuxExtractor, "use_aux": False},
        {"name": "5_StrategyGRU", "extractor": StrategyGRUAuxExtractor, "use_aux": True},
    ]

    history_channels = INTERACTION_HISTORY_CHANNELS if args.history_mode == "interaction" else HISTORY_CHANNELS

    settings = {
        "hand_path": args.hand_path,
        "total_steps": args.steps,
        "n_envs": args.n_envs,
        "eval_freq": args.eval_freq,
        "n_eval_episodes": args.n_eval_episodes,
        "history_length": args.history_length,
        "history_mode": args.history_mode,
        "history_channels": history_channels,
        "future_horizon": args.future_horizon,
        "scripted_hand_sample_prob": args.scripted_hand_sample_prob,
        "methods": [exp["name"] for exp in experiments],
    }
    with open(os.path.join(base_log_dir, "ablation_settings.json"), "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

    print("=" * 60)
    print("Network ablation")
    print("=" * 60)
    print(f"Opponent hand: {args.hand_path or 'scripted hand'}")
    print(f"Scripted sample probability: {args.scripted_hand_sample_prob}")
    print(f"History: length={args.history_length}, mode={args.history_mode}")
    print(f"Output: {base_log_dir}")
    print("=" * 60)

    for exp in experiments:
        exp_name = exp["name"]
        print(f"\n{'=' * 60}")
        print(f"Starting ablation experiment: {exp_name}")
        print(f"{'=' * 60}\n")

        vec_env = SubprocVecEnv([
            make_robot_env(
                hand_model_paths=hand_model_paths,
                history_length=args.history_length,
                history_mode=args.history_mode,
                scripted_hand_sample_prob=args.scripted_hand_sample_prob,
            )
            for _ in range(args.n_envs)
        ])

        extractor_kwargs = {}
        if exp["extractor"] is StrategyGRUAuxExtractor:
            extractor_kwargs = {
                "future_horizon": args.future_horizon,
                "history_channels": history_channels,
                "history_length": args.history_length,
            }

        policy_kwargs = dict(
            net_arch=[256, 256, 256, 64],
            features_extractor_class=exp["extractor"],
            features_extractor_kwargs=extractor_kwargs,
            share_features_extractor=True,
        )

        save_path = os.path.join(base_log_dir, exp_name)
        tb_dir = os.path.join(base_log_dir, "tensorboard_logs")
        os.makedirs(save_path, exist_ok=True)

        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=1e-4,
            batch_size=512,
            n_steps=2048,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_dir,
        )

        ablation_cb = AblationMetricsCallback(
            save_path=save_path,
            hand_model_paths=hand_model_paths,
            history_length=args.history_length,
            history_mode=args.history_mode,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            scripted_hand_sample_prob=args.scripted_hand_sample_prob,
            verbose=1,
        )
        callbacks = [ablation_cb]

        if exp["use_aux"]:
            callbacks.append(PPOFutureAuxTrainingCallback(
                history_length=args.history_length,
                history_channels=history_channels,
                future_horizon=args.future_horizon,
                batch_size=512,
                aux_epochs=3,
                traj_weight=args.strategy_traj_weight,
                risk_weight=args.strategy_risk_weight,
                contrastive_weight=args.strategy_contrastive_weight,
                contrastive_temperature=args.strategy_contrastive_temperature,
            ))

        model.learn(
            total_timesteps=args.steps,
            callback=CallbackList(callbacks),
            tb_log_name=exp_name,
        )

        model.save(os.path.join(save_path, "final_model.zip"))
        model.save(os.path.join(save_path, "best_model.zip"))
        ablation_cb.save_results(save_path)
        vec_env.close()
        del model

    print("\n" + "=" * 60)
    print("All ablation experiments completed!")
    print(f"Results saved to: {base_log_dir}")
    print(f"View training curves: tensorboard --logdir {tb_dir}")
    print("=" * 60)


def parse_args():
    parser = argparse.ArgumentParser(description="Train network ablations against a fixed hand opponent.")
    parser.add_argument("--hand_path", default=DEFAULT_HAND_PATH)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--run_name", default="ablation_h10_hand")
    parser.add_argument("--steps", type=int, default=8_000_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--eval_freq", type=int, default=50_000)
    parser.add_argument("--n_eval_episodes", type=int, default=50)
    parser.add_argument("--history_length", type=int, default=16)
    parser.add_argument("--history_mode", choices=["motion", "interaction"], default="motion")
    parser.add_argument("--future_horizon", type=int, default=4)
    parser.add_argument("--strategy_traj_weight", type=float, default=0.1)
    parser.add_argument("--strategy_risk_weight", type=float, default=0.02)
    parser.add_argument("--strategy_contrastive_weight", type=float, default=0.0)
    parser.add_argument("--strategy_contrastive_temperature", type=float, default=0.1)
    parser.add_argument("--scripted_hand_sample_prob", type=float, default=0.2)
    return parser.parse_args()


if __name__ == "__main__":
    run_ablation_study(parse_args())
