from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.custom_env import RehabilitationEnv
from src.utils.apf_controller import ReactiveAPFController


RUN_DIR = ROOT / "logs" / "league_zpd35_55_noid_warm_entropy_10iter_r5m_h1m_gru_noaux"
OUT_DIR = ROOT / "manuscripts" / "current_league_zpd35_55_noid_warm_entropy_10iter_final" / "apf_table2_baseline"
H1_PATH = RUN_DIR / "iteration_1" / "hand" / "hand" / "best_model.zip"


@dataclass
class TrialRecord:
    trial_index: int
    trial_seed: int
    episodes: int
    reward_mean: float
    reward_std: float
    tiz_mean: float
    tiz_std: float
    zpd_coverage_mean: float
    zpd_coverage_std: float
    episode_length_mean: float
    episode_length_std: float
    catch_rate: float
    too_close_rate: float
    too_far_rate: float
    avg_distance_mean: float
    avg_distance_std: float


@dataclass
class SummaryRow:
    robot_name: str
    robot_path: str
    test_name: str
    test_type: str
    hand_path: str | None
    trials: int
    episodes_per_trial: int
    episodes: int
    max_steps: int
    reward_mean: float
    reward_std: float
    tiz_mean: float
    tiz_std: float
    zpd_coverage_mean: float
    zpd_coverage_std: float
    episode_length_mean: float
    episode_length_std: float
    catch_rate: float
    too_close_rate: float
    too_far_rate: float
    avg_distance_mean: float
    avg_distance_std: float


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def spawn_seed_plan(base_seed: int, trials: int, episodes_per_trial: int):
    root_seq = np.random.SeedSequence(base_seed)
    trial_sequences = root_seq.spawn(trials)
    plan = []
    for trial_sequence in trial_sequences:
        trial_seed = int(trial_sequence.generate_state(1, dtype=np.uint32)[0])
        episode_sequences = trial_sequence.spawn(episodes_per_trial)
        episode_seeds = [
            int(sequence.generate_state(1, dtype=np.uint32)[0])
            for sequence in episode_sequences
        ]
        plan.append((trial_seed, episode_seeds))
    return plan


def sample_mean_std(values: list[float]):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return mean, std


def build_env(test_type: str, hand_model: PPO | None, max_steps: int) -> RehabilitationEnv:
    env = RehabilitationEnv(
        training_mode="robot",
        hand_model=hand_model,
        history_length=16,
        history_mode="motion",
    )
    env.random_noise = False
    env.max_steps = int(max_steps)
    env.scripted_hand_sample_prob = 1.0 if test_type == "scripted" else 0.0
    return env


def run_trial(
    test_name: str,
    test_type: str,
    hand_model: PPO | None,
    trial_index: int,
    trial_seed: int,
    episode_seeds: list[int],
    max_steps: int,
    controller_config: dict,
):
    env = build_env(test_type, hand_model, max_steps)
    controller = ReactiveAPFController(**controller_config)

    rewards = []
    tiz_values = []
    zpd_values = []
    lengths = []
    catches = []
    too_close_values = []
    too_far_values = []
    avg_distances = []

    try:
        for episode_seed in episode_seeds:
            seed_everything(episode_seed)
            obs, _ = env.reset(seed=episode_seed)
            controller.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            distances = []
            done_reason = ""

            while not (terminated or truncated):
                action = controller.predict(env)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                distances.append(float(info.get("dist", 0.0)))
                done_reason = str(info.get("done_reason", ""))

            dist_arr = np.asarray(distances, dtype=np.float64)
            if dist_arr.size:
                in_zpd = (dist_arr >= float(env.zpd_min)) & (dist_arr <= float(env.zpd_max))
                tiz_values.append(float(np.sum(in_zpd) / max_steps))
                zpd_values.append(float(np.mean(in_zpd)))
                too_close_values.append(float(np.mean(dist_arr < float(env.zpd_min))))
                too_far_values.append(float(np.mean(dist_arr > float(env.zpd_max))))
                avg_distances.append(float(np.mean(dist_arr)))
            else:
                tiz_values.append(0.0)
                zpd_values.append(0.0)
                too_close_values.append(0.0)
                too_far_values.append(0.0)
                avg_distances.append(0.0)
            rewards.append(total_reward)
            lengths.append(len(distances))
            catches.append(done_reason == "Robot Caught")
    finally:
        env.close()

    reward_mean, reward_std = sample_mean_std(rewards)
    tiz_mean, tiz_std = sample_mean_std(tiz_values)
    zpd_mean, zpd_std = sample_mean_std(zpd_values)
    length_mean, length_std = sample_mean_std([float(v) for v in lengths])
    avg_distance_mean, avg_distance_std = sample_mean_std(avg_distances)

    return TrialRecord(
        trial_index=int(trial_index),
        trial_seed=int(trial_seed),
        episodes=len(episode_seeds),
        reward_mean=reward_mean,
        reward_std=reward_std,
        tiz_mean=tiz_mean,
        tiz_std=tiz_std,
        zpd_coverage_mean=zpd_mean,
        zpd_coverage_std=zpd_std,
        episode_length_mean=length_mean,
        episode_length_std=length_std,
        catch_rate=float(np.mean(catches)) if catches else 0.0,
        too_close_rate=float(np.mean(too_close_values)) if too_close_values else 0.0,
        too_far_rate=float(np.mean(too_far_values)) if too_far_values else 0.0,
        avg_distance_mean=avg_distance_mean,
        avg_distance_std=avg_distance_std,
    )


def summarize_condition(test_name: str, test_type: str, hand_path: Path | None, max_steps: int, trials: list[TrialRecord]):
    def trial_stat(field: str):
        return sample_mean_std([float(getattr(t, field)) for t in trials])

    reward_mean, reward_std = trial_stat("reward_mean")
    tiz_mean, tiz_std = trial_stat("tiz_mean")
    zpd_mean, zpd_std = trial_stat("zpd_coverage_mean")
    length_mean, length_std = trial_stat("episode_length_mean")
    avg_distance_mean, avg_distance_std = trial_stat("avg_distance_mean")

    return SummaryRow(
        robot_name="reactive_apf",
        robot_path="ReactiveAPFController(radial+tangent+boundary)",
        test_name=test_name,
        test_type=test_type,
        hand_path=str(hand_path) if hand_path is not None else None,
        trials=len(trials),
        episodes_per_trial=trials[0].episodes if trials else 0,
        episodes=sum(t.episodes for t in trials),
        max_steps=int(max_steps),
        reward_mean=reward_mean,
        reward_std=reward_std,
        tiz_mean=tiz_mean,
        tiz_std=tiz_std,
        zpd_coverage_mean=zpd_mean,
        zpd_coverage_std=zpd_std,
        episode_length_mean=length_mean,
        episode_length_std=length_std,
        catch_rate=float(np.mean([t.catch_rate for t in trials])) if trials else 0.0,
        too_close_rate=float(np.mean([t.too_close_rate for t in trials])) if trials else 0.0,
        too_far_rate=float(np.mean([t.too_far_rate for t in trials])) if trials else 0.0,
        avg_distance_mean=avg_distance_mean,
        avg_distance_std=avg_distance_std,
    )


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate reactive APF baseline for Table II automatic conditions.")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--episodes_per_trial", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--h1_path", type=Path, default=H1_PATH)
    parser.add_argument("--radial_gain", type=float, default=1.20)
    parser.add_argument("--tangent_gain", type=float, default=0.30)
    parser.add_argument("--boundary_gain", type=float, default=1.35)
    parser.add_argument("--boundary_band", type=float, default=1.25)
    parser.add_argument("--boundary_guard", type=float, default=0.20)
    parser.add_argument("--smoothing", type=float, default=0.10)
    args = parser.parse_args()

    if args.trials < 1 or args.episodes_per_trial < 1:
        raise ValueError("trials and episodes_per_trial must be positive")
    if not args.h1_path.exists():
        raise FileNotFoundError(args.h1_path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    controller_config = {
        "radial_gain": args.radial_gain,
        "tangent_gain": args.tangent_gain,
        "boundary_gain": args.boundary_gain,
        "boundary_band": args.boundary_band,
        "boundary_guard": args.boundary_guard,
        "smoothing": args.smoothing,
    }

    tests = [
        ("scripted_hand", "scripted", None, None),
        ("agent_H1", "learned", args.h1_path, PPO.load(str(args.h1_path), custom_objects={"learning_rate": 0.0, "optimizer_class": None}, verbose=0)),
    ]

    summaries: list[SummaryRow] = []
    all_trials: list[dict] = []
    seed_plan = spawn_seed_plan(args.seed, args.trials, args.episodes_per_trial)

    for test_name, test_type, hand_path, hand_model in tests:
        print(f"\nEvaluating reactive_apf vs {test_name}")
        trial_records = []
        for trial_index, (trial_seed, episode_seeds) in enumerate(seed_plan):
            record = run_trial(
                test_name=test_name,
                test_type=test_type,
                hand_model=hand_model,
                trial_index=trial_index,
                trial_seed=trial_seed,
                episode_seeds=episode_seeds,
                max_steps=args.max_steps,
                controller_config=controller_config,
            )
            trial_records.append(record)
            row = asdict(record)
            row.update({"test_name": test_name, "test_type": test_type})
            all_trials.append(row)
            print(f"  Trial {trial_index + 1}/{args.trials}: TIZ={record.tiz_mean:.3f} | Len={record.episode_length_mean:.1f} | Catch={record.catch_rate:.2f}")
        summary = summarize_condition(test_name, test_type, hand_path, args.max_steps, trial_records)
        summaries.append(summary)
        print(f"  Summary: TIZ={summary.tiz_mean:.3f} +/- {summary.tiz_std:.3f} | Len={summary.episode_length_mean:.1f} +/- {summary.episode_length_std:.1f}")

    summary_rows = [asdict(row) for row in summaries]
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "run_dir": str(RUN_DIR),
        "base_seed": args.seed,
        "seed_strategy": "numpy.SeedSequence.spawn shared across test conditions",
        "controller": "ReactiveAPFController",
        "controller_config": controller_config,
        "trials": args.trials,
        "episodes_per_trial": args.episodes_per_trial,
        "max_steps": args.max_steps,
        "results": summary_rows,
        "trial_results": all_trials,
    }

    json_path = args.out_dir / f"apf_table2_results_{run_id}.json"
    csv_path = args.out_dir / f"apf_table2_results_{run_id}.csv"
    trials_path = args.out_dir / f"apf_table2_trials_{run_id}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(csv_path, summary_rows)
    write_csv(trials_path, all_trials)

    print(f"\nSaved JSON:   {json_path}")
    print(f"Saved summary:{csv_path}")
    print(f"Saved trials: {trials_path}")


if __name__ == "__main__":
    main()
