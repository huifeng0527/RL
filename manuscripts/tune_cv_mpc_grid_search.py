from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path

from stable_baselines3 import PPO

from run_cv_mpc_table2_baseline import (
    H1_PATH,
    OUT_DIR,
    RUN_DIR,
    parse_float_list,
    run_trial,
    spawn_seed_plan,
    summarize_condition,
)


def parse_int_list(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_config(
    controller_config: dict,
    seed_plan,
    h1_path: Path,
    h1_model: PPO,
    max_steps: int,
    diagnostic_boundary_band: float,
):
    tests = [
        ("scripted_hand", "scripted", None, None),
        ("agent_H1", "learned", h1_path, h1_model),
    ]
    summaries = []
    all_trials = []
    for test_name, test_type, hand_path, hand_model in tests:
        trial_records = []
        for trial_index, (trial_seed, episode_seeds) in enumerate(seed_plan):
            record = run_trial(
                test_name=test_name,
                test_type=test_type,
                hand_model=hand_model,
                trial_index=trial_index,
                trial_seed=trial_seed,
                episode_seeds=episode_seeds,
                max_steps=max_steps,
                controller_config=controller_config,
                diagnostic_boundary_band=diagnostic_boundary_band,
            )
            trial_records.append(record)
            trial_row = asdict(record)
            trial_row.update({"test_name": test_name, "test_type": test_type})
            all_trials.append(trial_row)
        summaries.append(summarize_condition(test_name, test_type, hand_path, max_steps, trial_records))
    return summaries, all_trials


def flatten_summary(prefix: str, summary) -> dict:
    return {
        f"{prefix}_tiz_mean": summary.tiz_mean,
        f"{prefix}_tiz_std": summary.tiz_std,
        f"{prefix}_episode_length_mean": summary.episode_length_mean,
        f"{prefix}_episode_length_std": summary.episode_length_std,
        f"{prefix}_catch_rate": summary.catch_rate,
        f"{prefix}_oob_rate": summary.oob_rate,
        f"{prefix}_boundary_occupancy_mean": summary.boundary_occupancy_mean,
        f"{prefix}_min_clearance_mean": summary.min_clearance_mean,
        f"{prefix}_min_clearance_p05": summary.min_clearance_p05,
        f"{prefix}_too_close_rate": summary.too_close_rate,
        f"{prefix}_too_far_rate": summary.too_far_rate,
        f"{prefix}_avg_distance_mean": summary.avg_distance_mean,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Tune the two-move sequence constant-velocity MPC baseline on validation-only seeds."
    )
    parser.add_argument("--validation_trials", type=int, default=3)
    parser.add_argument("--validation_episodes_per_trial", type=int, default=100)
    parser.add_argument("--final_trials", type=int, default=20)
    parser.add_argument("--final_episodes_per_trial", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--validation_seed", type=int, default=32026)
    parser.add_argument("--final_seed", type=int, default=2026)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=OUT_DIR / "sequence_cv_mpc_grid_search_20260804",
    )
    parser.add_argument("--h1_path", type=Path, default=H1_PATH)
    parser.add_argument("--horizon_grid", default="4,5,6")
    parser.add_argument("--velocity_window_grid", default="3,5,7")
    parser.add_argument("--boundary_barrier_weight_grid", default="8,12,16")
    parser.add_argument("--collision_barrier_weight_grid", default="4,8,16")
    parser.add_argument("--action_grid", default="-1,-0.5,0,0.5,1")
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--effort_weight", type=float, default=0.01)
    parser.add_argument("--smoothness_weight", type=float, default=0.03)
    parser.add_argument("--collision_penalty", type=float, default=120.0)
    parser.add_argument("--oob_penalty", type=float, default=240.0)
    parser.add_argument("--boundary_band", type=float, default=1.0)
    parser.add_argument("--collision_buffer", type=float, default=1.0)
    parser.add_argument("--diagnostic_boundary_band", type=float, default=0.5)
    parser.add_argument("--max_boundary_occupancy", type=float, default=0.10)
    parser.add_argument("--min_clearance_p05", type=float, default=0.05)
    parser.add_argument("--skip_final", action="store_true")
    args = parser.parse_args()

    if not args.h1_path.exists():
        raise FileNotFoundError(args.h1_path)
    if args.validation_seed == args.final_seed:
        raise ValueError("validation_seed and final_seed must be different")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")

    horizon_values = parse_int_list(args.horizon_grid)
    velocity_window_values = parse_int_list(args.velocity_window_grid)
    boundary_weight_values = parse_float_list(args.boundary_barrier_weight_grid)
    collision_weight_values = parse_float_list(args.collision_barrier_weight_grid)
    action_grid = parse_float_list(args.action_grid)
    grid = list(itertools.product(
        horizon_values,
        velocity_window_values,
        boundary_weight_values,
        collision_weight_values,
    ))
    validation_seed_plan = spawn_seed_plan(
        args.validation_seed,
        args.validation_trials,
        args.validation_episodes_per_trial,
    )

    h1_model = PPO.load(
        str(args.h1_path),
        custom_objects={"learning_rate": 0.0, "optimizer_class": None},
        verbose=0,
    )

    print(f"Sequence CV-MPC grid search run_id={run_id}")
    print(f"Grid size: {len(grid)} configurations")
    print(
        f"Validation: {args.validation_trials} trials x "
        f"{args.validation_episodes_per_trial} episodes per condition, "
        f"seed={args.validation_seed}"
    )

    grid_csv_path = args.out_dir / f"sequence_cv_mpc_grid_search_results_{run_id}.csv"
    grid_rows: list[dict] = []

    for index, (horizon, velocity_window, boundary_weight, collision_weight) in enumerate(grid, start=1):
        controller_config = {
            "horizon": horizon,
            "velocity_window": velocity_window,
            "action_grid": action_grid,
            "discount": args.discount,
            "effort_weight": args.effort_weight,
            "smoothness_weight": args.smoothness_weight,
            "collision_penalty": args.collision_penalty,
            "oob_penalty": args.oob_penalty,
            "boundary_band": args.boundary_band,
            "boundary_barrier_weight": boundary_weight,
            "collision_buffer": args.collision_buffer,
            "collision_barrier_weight": collision_weight,
        }
        summaries, _ = evaluate_config(
            controller_config,
            validation_seed_plan,
            args.h1_path,
            h1_model,
            args.max_steps,
            args.diagnostic_boundary_band,
        )
        by_test = {summary.test_name: summary for summary in summaries}
        scripted = by_test["scripted_hand"]
        h1 = by_test["agent_H1"]
        score_mean_tiz = 0.5 * (scripted.tiz_mean + h1.tiz_mean)
        score_min_tiz = min(scripted.tiz_mean, h1.tiz_mean)
        mean_catch_rate = 0.5 * (scripted.catch_rate + h1.catch_rate)
        mean_boundary_occupancy = 0.5 * (
            scripted.boundary_occupancy_mean + h1.boundary_occupancy_mean
        )
        safety_pass = bool(
            scripted.oob_rate == 0.0
            and h1.oob_rate == 0.0
            and scripted.boundary_occupancy_mean <= args.max_boundary_occupancy
            and h1.boundary_occupancy_mean <= args.max_boundary_occupancy
            and scripted.min_clearance_p05 >= args.min_clearance_p05
            and h1.min_clearance_p05 >= args.min_clearance_p05
        )
        row = {
            "rank_input_order": index,
            **controller_config,
            "action_grid": ",".join(str(v) for v in action_grid),
            "safety_pass": safety_pass,
            "score_mean_tiz": score_mean_tiz,
            "score_min_tiz": score_min_tiz,
            "mean_catch_rate": mean_catch_rate,
            "mean_boundary_occupancy": mean_boundary_occupancy,
            **flatten_summary("scripted", scripted),
            **flatten_summary("h1", h1),
        }
        grid_rows.append(row)
        write_csv(grid_csv_path, grid_rows)
        print(
            f"[{index:03d}/{len(grid):03d}] h={horizon} vw={velocity_window} "
            f"bw={boundary_weight:g} cw={collision_weight:g} | "
            f"safe={int(safety_pass)} score={score_mean_tiz:.4f} min={score_min_tiz:.4f} | "
            f"SPC={scripted.tiz_mean:.4f} H1={h1.tiz_mean:.4f} | "
            f"OOB={scripted.oob_rate:.3f}/{h1.oob_rate:.3f} "
            f"Boundary={scripted.boundary_occupancy_mean:.3f}/{h1.boundary_occupancy_mean:.3f}"
        )

    safe_rows = [row for row in grid_rows if row["safety_pass"]]
    if not safe_rows:
        summary_json_path = args.out_dir / f"sequence_cv_mpc_grid_search_summary_{run_id}.json"
        payload = {
            "schema_version": 2,
            "run_id": run_id,
            "validation_seed": args.validation_seed,
            "final_seed": args.final_seed,
            "selection_status": "no_safe_configuration",
            "grid_results": grid_rows,
        }
        summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise RuntimeError(
            "No CV-MPC configuration passed the safety diagnostics. "
            f"Inspect {summary_json_path} and adjust the controller/search bounds."
        )

    ranked = sorted(
        safe_rows,
        key=lambda row: (
            row["score_mean_tiz"],
            row["score_min_tiz"],
            row["scripted_episode_length_mean"] + row["h1_episode_length_mean"],
            -row["mean_catch_rate"],
            -row["mean_boundary_occupancy"],
        ),
        reverse=True,
    )
    selected = ranked[0]
    selected_config = {
        "horizon": int(selected["horizon"]),
        "velocity_window": int(selected["velocity_window"]),
        "action_grid": action_grid,
        "discount": args.discount,
        "effort_weight": args.effort_weight,
        "smoothness_weight": args.smoothness_weight,
        "collision_penalty": args.collision_penalty,
        "oob_penalty": args.oob_penalty,
        "boundary_band": args.boundary_band,
        "boundary_barrier_weight": float(selected["boundary_barrier_weight"]),
        "collision_buffer": args.collision_buffer,
        "collision_barrier_weight": float(selected["collision_barrier_weight"]),
    }

    print("\nTop 10 safe validation configurations:")
    for rank, row in enumerate(ranked[:10], start=1):
        print(
            f"#{rank}: score={row['score_mean_tiz']:.4f}, min={row['score_min_tiz']:.4f}, "
            f"SPC={row['scripted_tiz_mean']:.4f}, H1={row['h1_tiz_mean']:.4f}, "
            f"catch={row['mean_catch_rate']:.3f}, boundary={row['mean_boundary_occupancy']:.3f}, "
            f"config={{h={row['horizon']}, vw={row['velocity_window']}, "
            f"bw={row['boundary_barrier_weight']}, cw={row['collision_barrier_weight']}}}"
        )

    final_payload = None
    if not args.skip_final:
        print(f"\nRunning held-out final retest with seed={args.final_seed}...")
        final_seed_plan = spawn_seed_plan(
            args.final_seed,
            args.final_trials,
            args.final_episodes_per_trial,
        )
        final_summaries, final_trials = evaluate_config(
            selected_config,
            final_seed_plan,
            args.h1_path,
            h1_model,
            args.max_steps,
            args.diagnostic_boundary_band,
        )
        final_summary_rows = [asdict(summary) for summary in final_summaries]
        for row in final_summary_rows:
            row["robot_name"] = "sequence_cv_mpc"
            row["robot_path"] = (
                "ConstantVelocityMPCController(two-move sequence shooting, "
                "constant-hand-velocity)"
            )
        final_payload = {
            "schema_version": 2,
            "run_id": run_id,
            "run_dir": str(RUN_DIR),
            "validation_seed": args.validation_seed,
            "final_seed": args.final_seed,
            "seed_strategy": "separate numpy.SeedSequence.spawn plans for validation and held-out final evaluation",
            "controller": "TwoMoveSequenceConstantVelocityMPCController",
            "controller_config": selected_config,
            "prediction_model": "Observed Hand velocity held constant; all ordered two-action move-blocking sequences are scored.",
            "privileged_information_used": False,
            "diagnostic_boundary_band": args.diagnostic_boundary_band,
            "final_trials": args.final_trials,
            "final_episodes_per_trial": args.final_episodes_per_trial,
            "max_steps": args.max_steps,
            "results": final_summary_rows,
            "trial_results": final_trials,
        }
        final_json_path = args.out_dir / f"sequence_cv_mpc_table2_results_{run_id}.json"
        final_csv_path = args.out_dir / f"sequence_cv_mpc_table2_results_{run_id}.csv"
        final_trials_path = args.out_dir / f"sequence_cv_mpc_table2_trials_{run_id}.csv"
        final_json_path.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")
        write_csv(final_csv_path, final_summary_rows)
        write_csv(final_trials_path, final_trials)
        for summary in final_summaries:
            print(
                f"  Final {summary.test_name}: TIZ={summary.tiz_mean:.4f} +/- {summary.tiz_std:.4f} | "
                f"Len={summary.episode_length_mean:.2f} +/- {summary.episode_length_std:.2f} | "
                f"Catch={summary.catch_rate:.3f} OOB={summary.oob_rate:.3f} "
                f"Boundary={summary.boundary_occupancy_mean:.3f}"
            )
        print(f"Saved final JSON: {final_json_path}")
        print(f"Saved final CSV:  {final_csv_path}")
        print(f"Saved final trials: {final_trials_path}")

    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "run_dir": str(RUN_DIR),
        "validation_seed": args.validation_seed,
        "final_seed": args.final_seed,
        "h1_path": str(args.h1_path),
        "validation_trials": args.validation_trials,
        "validation_episodes_per_trial": args.validation_episodes_per_trial,
        "max_steps": args.max_steps,
        "selection_metric": (
            "reject OOB/high-boundary-occupancy configurations, then maximize mean and minimum "
            "validation TIZ with episode length, catch rate, and boundary occupancy tie-breakers"
        ),
        "safety_thresholds": {
            "diagnostic_boundary_band": args.diagnostic_boundary_band,
            "max_boundary_occupancy": args.max_boundary_occupancy,
            "min_clearance_p05": args.min_clearance_p05,
            "max_oob_rate": 0.0,
        },
        "grid": {
            "horizon": horizon_values,
            "velocity_window": velocity_window_values,
            "boundary_barrier_weight": boundary_weight_values,
            "collision_barrier_weight": collision_weight_values,
            "action_grid": action_grid,
            "discount": args.discount,
            "effort_weight": args.effort_weight,
            "smoothness_weight": args.smoothness_weight,
            "collision_penalty": args.collision_penalty,
            "oob_penalty": args.oob_penalty,
            "boundary_band": args.boundary_band,
            "collision_buffer": args.collision_buffer,
        },
        "selected_config": selected_config,
        "top_10": ranked[:10],
        "grid_results": grid_rows,
        "final_results": final_payload,
    }
    summary_json_path = args.out_dir / f"sequence_cv_mpc_grid_search_summary_{run_id}.json"
    summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved grid CSV: {grid_csv_path}")
    print(f"Saved grid summary JSON: {summary_json_path}")


if __name__ == "__main__":
    main()
