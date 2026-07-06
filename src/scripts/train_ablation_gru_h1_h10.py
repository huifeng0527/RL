"""Ablation: MLP vs GRU vs GRU+Aux against h1-h10 + scripted hand pool.

Single-generation training.  Three groups:
  1. MLPOnlyExtractor            (no history, no aux)
  2. StrategyGRUAuxExtractor     (GRU strategy encoder, no aux loss)
  3. StrategyGRUAuxExtractor     (GRU strategy encoder + multi-risk aux loss)

Usage:
    python src/scripts/train_ablation_gru_h1_h10.py
    python src/scripts/train_ablation_gru_h1_h10.py --steps 3000000
"""

import argparse
import datetime
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.custom_env import RehabilitationEnv
from src.observation_schema import INTERACTION_HISTORY_CHANNELS
from src.utils.ablation_callbacks import PPOFutureAuxTrainingCallback
from src.utils.feature_extractors import MLPOnlyExtractor, StrategyGRUAuxExtractor

DEFAULT_LEAGUE_DIR = (
    "logs/league_zpd35_55_noid_warm_entropy_10iter_r5m_h1m_gru_noaux"
)

# env workspace bounds (from RehabilitationEnv defaults)
ENV_WIDTH = 15.0
ENV_HEIGHT = 10.0
MARGIN = 0.6
MAX_WORKSPACE_AREA = (ENV_WIDTH - 2 * MARGIN) * (ENV_HEIGHT - 2 * MARGIN)


# ── Per-episode training metrics ──────────────────────────────────────────────

class EpisodeMetricsCallback(BaseCallback):
    """Record per-episode zpd_coverage, workspace_area, reward, length during training."""

    def __init__(self, save_path, window=200, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.window = window
        # raw per-episode data
        self.timesteps = []
        self.rewards = []
        self.ep_lengths = []
        self.zpd_coverages = []
        self.zpd_steps = []
        self.workspace_areas = []
        # per-env accumulators
        self._robot_xs = None
        self._robot_ys = None

    def _on_training_start(self):
        existing_path = os.path.join(self.save_path, "episode_metrics.npz")
        if os.path.exists(existing_path):
            existing = np.load(existing_path)
            self.timesteps = existing["timesteps"].astype(np.int64).tolist()
            self.rewards = existing["rewards"].astype(np.float32).tolist()
            self.ep_lengths = existing["ep_lengths"].astype(np.int64).tolist()
            self.zpd_coverages = existing["zpd_coverages"].astype(np.float32).tolist()
            if "zpd_steps" in existing.files:
                self.zpd_steps = existing["zpd_steps"].astype(np.float32).tolist()
            else:
                self.zpd_steps = (existing["zpd_coverages"] * existing["ep_lengths"]).astype(np.float32).tolist()
            self.workspace_areas = existing["workspace_areas"].astype(np.float32).tolist()
            if self.verbose > 0:
                print(f"Loaded {len(self.timesteps)} existing episode metrics from {existing_path}")

        n = self.training_env.num_envs
        self._robot_xs = [[] for _ in range(n)]
        self._robot_ys = [[] for _ in range(n)]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, info in enumerate(infos):
            # accumulate robot positions each step
            rp = info.get("robot_pos")
            if rp is not None:
                self._robot_xs[i].append(float(rp[0]))
                self._robot_ys[i].append(float(rp[1]))

            # on episode end, record metrics
            if i < len(dones) and dones[i]:
                ep = info.get("league_episode")
                if ep is not None:
                    self.timesteps.append(self.num_timesteps)
                    ep_length = int(ep.get("episode_length", 0))
                    zpd_coverage = float(ep.get("zpd_coverage", 0.0))
                    self.rewards.append(float(ep.get("reward", 0.0)))
                    self.ep_lengths.append(ep_length)
                    self.zpd_coverages.append(zpd_coverage)
                    self.zpd_steps.append(zpd_coverage * ep_length)

                    xs = self._robot_xs[i]
                    ys = self._robot_ys[i]
                    if len(xs) > 1:
                        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                        self.workspace_areas.append(area / MAX_WORKSPACE_AREA)
                    else:
                        self.workspace_areas.append(0.0)
                # reset accumulators for this env
                self._robot_xs[i] = []
                self._robot_ys[i] = []
        return True

    def _on_training_end(self):
        self._save()

    def _save(self):
        os.makedirs(self.save_path, exist_ok=True)
        np.savez(
            os.path.join(self.save_path, "episode_metrics.npz"),
            timesteps=np.array(self.timesteps, dtype=np.int64),
            rewards=np.array(self.rewards, dtype=np.float32),
            ep_lengths=np.array(self.ep_lengths, dtype=np.int64),
            zpd_coverages=np.array(self.zpd_coverages, dtype=np.float32),
            zpd_steps=np.array(self.zpd_steps, dtype=np.float32),
            workspace_areas=np.array(self.workspace_areas, dtype=np.float32),
        )


# ── Periodic eval (deterministic policy) ─────────────────────────────────────

class AblationMetricsCallback(BaseCallback):
    """Evaluate robot vs the full hand pool periodically."""

    def __init__(
        self,
        save_path,
        hand_model_paths,
        history_length=16,
        history_mode="interaction",
        eval_freq=10000,
        n_eval_episodes=50,
        scripted_hand_sample_prob=0.1,
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
        self.results = []
        self.best_reward = None

    def _on_training_start(self):
        metrics_path = os.path.join(self.save_path, "ablation_metrics.json")
        best_path = os.path.join(self.save_path, "ablation_best.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                self.results = json.load(f)
        if os.path.exists(best_path):
            with open(best_path, "r") as f:
                self.best_reward = json.load(f)
        if self.verbose > 0 and self.results:
            print(f"Loaded {len(self.results)} existing eval records from {metrics_path}")

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
        episode_lengths = []
        rewards = []

        for _ in range(self.n_eval_episodes):
            obs, _ = eval_env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            in_zpd_count = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                dist = info.get("dist", 0.0)
                if zpd_min <= dist <= zpd_max:
                    in_zpd_count += 1

            zpd_coverages.append(in_zpd_count / steps if steps > 0 else 0.0)
            episode_lengths.append(steps)
            rewards.append(total_reward)

        eval_env.close()

        result = {
            "timestep": self.num_timesteps,
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "episode_length_mean": float(np.mean(episode_lengths)),
            "zpd_coverage_mean": float(np.mean(zpd_coverages)),
            "zpd_coverage_std": float(np.std(zpd_coverages)),
        }
        self.results.append(result)
        if self.best_reward is None or result["reward_mean"] > self.best_reward["reward_mean"]:
            self.best_reward = result
        self.save_results()

        self.logger.record("eval/reward_mean", result["reward_mean"])
        self.logger.record("eval/zpd_coverage_mean", result["zpd_coverage_mean"])
        self.logger.record("eval/episode_length_mean", result["episode_length_mean"])

        if self.verbose > 0:
            print(
                f"\n[Eval {self.num_timesteps}] R: {result['reward_mean']:.2f} | "
                f"Len: {result['episode_length_mean']:.1f} | "
                f"ZPD: {result['zpd_coverage_mean']:.1%}"
            )
        return True

    def save_results(self):
        os.makedirs(self.save_path, exist_ok=True)
        with open(os.path.join(self.save_path, "ablation_metrics.json"), "w") as f:
            json.dump(self.results, f, indent=2)
        with open(os.path.join(self.save_path, "ablation_best.json"), "w") as f:
            json.dump(self.best_reward, f, indent=2)


# ── Plotting ─────────────────────────────────────────────────────────────────

def smooth(y, window=500):
    """Moving average smoothing."""
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")


def plot_training_curves(out_dir, all_data):
    """Plot zpd_coverage and workspace_area training curves for all groups.

    all_data: dict  {exp_name: npz_dict}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"1_MLP": "MLP", "2_GRU_Seq": "GRU (no aux)", "3_GRU_Aux": "GRU + aux"}
    colors = {"1_MLP": "#4C72B0", "2_GRU_Seq": "#DD8452", "3_GRU_Aux": "#55A868"}
    smooth_window = 500

    for metric_key, ylabel, fname in [
        ("zpd_steps", "Steps in ZPD", "curve_zpd_steps.png"),
        ("zpd_coverages", "ZPD Coverage", "curve_zpd_coverage.png"),
        ("workspace_areas", "Workspace Coverage", "curve_workspace_coverage.png"),
        ("rewards", "Episode Reward", "curve_reward.png"),
        ("ep_lengths", "Episode Length", "curve_episode_length.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for exp_name, data in all_data.items():
            ts = data["timesteps"].astype(np.float64) / 1e6  # to millions
            vals = data[metric_key].astype(np.float64)
            if len(vals) == 0:
                continue
            # raw (faded)
            ax.plot(ts, vals, color=colors.get(exp_name, "gray"), alpha=0.1, linewidth=0.5)
            # smoothed
            if len(vals) > smooth_window:
                s_vals = smooth(vals, smooth_window)
                s_ts = ts[smooth_window - 1:]
                ax.plot(s_ts, s_vals, color=colors.get(exp_name, "gray"),
                        label=labels.get(exp_name, exp_name), linewidth=1.5)
            else:
                ax.plot(ts, vals, color=colors.get(exp_name, "gray"),
                        label=labels.get(exp_name, exp_name), linewidth=1.5)

        ax.set_xlabel("Timesteps (M)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_final_bars(out_dir, all_data):
    """Bar chart: last 10% mean for each metric per group."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"1_MLP": "MLP", "2_GRU_Seq": "GRU (no aux)", "3_GRU_Aux": "GRU + aux"}
    colors = {"1_MLP": "#4C72B0", "2_GRU_Seq": "#DD8452", "3_GRU_Aux": "#55A868"}

    metrics = [
        ("zpd_steps", "Steps in ZPD"),
        ("zpd_coverages", "ZPD Coverage"),
        ("workspace_areas", "Workspace Coverage"),
        ("rewards", "Episode Reward"),
        ("ep_lengths", "Episode Length"),
    ]

    exp_names = list(all_data.keys())
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))

    for ax, (key, title) in zip(axes, metrics):
        means = []
        stds = []
        for en in exp_names:
            vals = all_data[en][key].astype(np.float64)
            if len(vals) > 0:
                tail = vals[max(0, int(len(vals) * 0.9)):]
                means.append(np.mean(tail))
                stds.append(np.std(tail))
            else:
                means.append(0.0)
                stds.append(0.0)

        x = np.arange(len(exp_names))
        bar_colors = [colors.get(en, "gray") for en in exp_names]
        ax.bar(x, means, yerr=stds, color=bar_colors, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([labels.get(en, en) for en in exp_names], rotation=15)
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bar_final_comparison.png"), dpi=150)
    plt.close(fig)
    print("  Saved bar_final_comparison.png")


# ── Helpers ──────────────────────────────────────────────────────────────────

def collect_hand_paths(league_dir):
    """Gather iteration_1..10 hand best_model.zip paths."""
    paths = []
    for i in range(1, 11):
        p = os.path.join(league_dir, f"iteration_{i}", "hand", "hand", "best_model.zip")
        if os.path.exists(p):
            paths.append(p)
        else:
            print(f"  [warn] missing: {p}")
    return paths


def make_robot_env(hand_model_paths, history_length, history_mode, scripted_prob):
    def _init():
        env = RehabilitationEnv(
            training_mode="robot",
            hand_model_paths=hand_model_paths,
            history_length=history_length,
            history_mode=history_mode,
        )
        env.scripted_hand_sample_prob = scripted_prob
        return Monitor(env)
    return _init


# ── Main ─────────────────────────────────────────────────────────────────────

def run(args):
    hand_paths = collect_hand_paths(args.league_dir)
    if not hand_paths:
        print("No hand models found. Aborting.")
        return
    print(f"Hand pool: {len(hand_paths)} models from {args.league_dir}")

    now = datetime.datetime.now().strftime("%m%d_%H%M")
    out = args.output_dir or f"logs/ablation_gru_h1_h10_{now}"
    os.makedirs(out, exist_ok=True)

    history_channels = INTERACTION_HISTORY_CHANNELS
    future_horizon = args.future_horizon

    experiments = [
        {
            "name": "1_MLP",
            "extractor": MLPOnlyExtractor,
            "extractor_kwargs": {},
            "use_aux": False,
        },
        {
            "name": "2_GRU_Seq",
            "extractor": StrategyGRUAuxExtractor,
            "extractor_kwargs": {
                "future_horizon": future_horizon,
                "history_channels": history_channels,
                "history_length": args.history_length,
            },
            "use_aux": False,
        },
        {
            "name": "3_GRU_Aux",
            "extractor": StrategyGRUAuxExtractor,
            "extractor_kwargs": {
                "future_horizon": future_horizon,
                "history_channels": history_channels,
                "history_length": args.history_length,
            },
            "use_aux": True,
        },
    ]

    selected_groups = set(args.groups)
    experiments = [exp for exp in experiments if exp["name"] in selected_groups]
    if not experiments:
        raise ValueError(f"No experiments selected from groups={args.groups}")

    cfg = {
        "league_dir": args.league_dir,
        "hand_paths": hand_paths,
        "steps": args.steps,
        "n_envs": args.n_envs,
        "history_length": args.history_length,
        "history_mode": args.history_mode,
        "future_horizon": future_horizon,
        "scripted_hand_sample_prob": args.scripted_hand_sample_prob,
        "traj_weight": args.traj_weight,
        "risk_weight": args.risk_weight,
        "contrastive_weight": args.contrastive_weight,
        "traj_cumulative_weight": args.traj_cumulative_weight,
        "traj_endpoint_weight": args.traj_endpoint_weight,
        "requested_groups": args.groups,
        "load_model": args.load_model,
        "resume_timesteps": args.resume_timesteps,
        "experiments": [e["name"] for e in experiments],
    }
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    print("=" * 60)
    print("GRU Ablation: MLP / GRU_Seq / GRU_Aux")
    print(f"  Steps per group: {args.steps:,}")
    print(f"  Envs: {args.n_envs}")
    print(f"  History: len={args.history_length}, mode={args.history_mode}")
    print(f"  Future horizon: {future_horizon}")
    print(f"  Scripted prob: {args.scripted_hand_sample_prob}")
    print(f"  Output: {out}")
    print("=" * 60)

    for exp in experiments:
        name = exp["name"]
        print(f"\n{'=' * 60}")
        print(f"  {name}")
        print(f"{'=' * 60}\n")

        save_dir = os.path.join(out, name)
        os.makedirs(save_dir, exist_ok=True)

        vec_env = SubprocVecEnv([
            make_robot_env(hand_paths, args.history_length, args.history_mode, args.scripted_hand_sample_prob)
            for _ in range(args.n_envs)
        ])

        policy_kwargs = dict(
            net_arch=[256, 256, 256, 64],
            features_extractor_class=exp["extractor"],
            features_extractor_kwargs=exp["extractor_kwargs"],
            share_features_extractor=True,
        )

        if args.load_model:
            if len(experiments) != 1:
                raise ValueError("--load_model is only supported when running one group")
            print(f"Loading model from: {args.load_model}")
            model = PPO.load(
                args.load_model,
                env=vec_env,
                verbose=1,
                tensorboard_log=os.path.join(out, "tb"),
            )
        else:
            model = PPO(
                "MlpPolicy",
                vec_env,
                verbose=1,
                learning_rate=1e-4,
                batch_size=512,
                n_steps=2048,
                policy_kwargs=policy_kwargs,
                tensorboard_log=os.path.join(out, "tb"),
            )

        callbacks = [
            EpisodeMetricsCallback(save_path=save_dir, verbose=1),
            AblationMetricsCallback(
                save_path=save_dir,
                hand_model_paths=hand_paths,
                history_length=args.history_length,
                history_mode=args.history_mode,
                eval_freq=args.eval_freq,
                n_eval_episodes=args.n_eval_episodes,
                scripted_hand_sample_prob=args.scripted_hand_sample_prob,
                verbose=1,
            ),
        ]

        if exp["use_aux"]:
            callbacks.append(PPOFutureAuxTrainingCallback(
                history_length=args.history_length,
                history_channels=history_channels,
                future_horizon=future_horizon,
                batch_size=512,
                aux_epochs=3,
                traj_weight=args.traj_weight,
                risk_weight=args.risk_weight,
                contrastive_weight=args.contrastive_weight,
                traj_cumulative_weight=args.traj_cumulative_weight,
                traj_endpoint_weight=args.traj_endpoint_weight,
                verbose=1,
            ))

        model.learn(
            total_timesteps=args.steps,
            callback=CallbackList(callbacks),
            tb_log_name=name,
            reset_num_timesteps=not args.resume_timesteps,
        )

        model.save(os.path.join(save_dir, "final_model.zip"))
        model.save(os.path.join(save_dir, "best_model.zip"))
        vec_env.close()
        del model

    # ── Plot all groups ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Generating figures...")

    all_data = {}
    for exp in experiments:
        npz_path = os.path.join(out, exp["name"], "episode_metrics.npz")
        if os.path.exists(npz_path):
            all_data[exp["name"]] = dict(np.load(npz_path, allow_pickle=True))
            print(f"  Loaded {exp['name']}: {len(all_data[exp['name']]['timesteps'])} episodes")

    if all_data:
        plot_training_curves(out, all_data)
        plot_final_bars(out, all_data)
    else:
        print("  No data to plot.")

    print(f"\nDone. All results in: {out}")


def main():
    p = argparse.ArgumentParser(description="GRU ablation against h1-h10 hand pool.")
    p.add_argument("--league_dir", default=DEFAULT_LEAGUE_DIR)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--n_envs", type=int, default=4)
    p.add_argument("--eval_freq", type=int, default=50_000)
    p.add_argument("--n_eval_episodes", type=int, default=50)
    p.add_argument("--history_length", type=int, default=16)
    p.add_argument("--history_mode", choices=["motion", "interaction"], default="interaction")
    p.add_argument("--future_horizon", type=int, default=8)
    p.add_argument("--scripted_hand_sample_prob", type=float, default=0.1)
    p.add_argument("--traj_weight", type=float, default=0.1)
    p.add_argument("--risk_weight", type=float, default=0.02)
    p.add_argument("--contrastive_weight", type=float, default=0.0)
    p.add_argument("--traj_cumulative_weight", type=float, default=0.0)
    p.add_argument("--traj_endpoint_weight", type=float, default=0.0)
    p.add_argument("--load_model", default=None)
    p.add_argument("--resume_timesteps", action="store_true")
    p.add_argument(
        "--groups",
        nargs="+",
        choices=["1_MLP", "2_GRU_Seq", "3_GRU_Aux"],
        default=["1_MLP", "2_GRU_Seq", "3_GRU_Aux"],
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
