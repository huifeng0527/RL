from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path

from stable_baselines3 import PPO

from run_apf_table2_baseline import (
    H1_PATH,
    OUT_DIR,
    RUN_DIR,
    run_trial,
    spawn_seed_plan,
    summarize_condition,
)


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


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
        f"{prefix}_too_close_rate": summary.too_close_rate,
        f"{prefix}_too_far_rate": summary.too_far_rate,
        f"{prefix}_avg_distance_mean": summary.avg_distance_mean,
    }


def main():
    parser = argparse.ArgumentParser(description="Grid-search APF parameters and retest the tuned APF for Table II automatic conditions.")
    parser.add_argument("--validation_trials", type=int, default=3)
    parser.add_argument("--validation_episodes_per_trial", type=int, default=200)
    parser.add_argument("--final_trials", type=int, default=20)
    parser.add_argument("--final_episodes_per_trial", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR / "apf_grid_search_20260730")
    parser.add_argument("--h1_path", type=Path, default=H1_PATH)
    parser.add_argument("--radial_grid", default="0.8,1.2,1.6")
    parser.add_argument("--tangent_grid", default="0.0,0.3,0.6")
    parser.add_argument("--boundary_gain_grid", default="0.8,1.35,2.0")
    parser.add_argument("--smoothing_grid", default="0.0,0.1")
    parser.add_argument("--boundary_band", type=float, default=1.25)
    parser.add_argument("--boundary_guard", type=float, default=0.20)
    parser.add_argument("--skip_final", action="store_true")
    args = parser.parse_args()

    if not args.h1_path.exists():
        raise FileNotFoundError(args.h1_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")

    radial_values = parse_float_list(args.radial_grid)
    tangent_values = parse_float_list(args.tangent_grid)
    boundary_values = parse_float_list(args.boundary_gain_grid)
    smoothing_values = parse_float_list(args.smoothing_grid)
    grid = list(itertools.product(radial_values, tangent_values, boundary_values, smoothing_values))
    validation_seed_plan = spawn_seed_plan(args.seed, args.validation_trials, args.validation_episodes_per_trial)

    h1_model = PPO.load(
        str(args.h1_path),
        custom_objects={"learning_rate": 0.0, "optimizer_class": None},
        verbose=0,
    )

    print(f"APF grid search run_id={run_id}")
    print(f"Grid size: {len(grid)} configurations")
    print(f"Validation: {args.validation_trials} trials x {args.validation_episodes_per_trial} episodes per condition")

    grid_csv_path = args.out_dir / f"apf_grid_search_results_{run_id}.csv"
    grid_rows: list[dict] = []

    for index, (radial_gain, tangent_gain, boundary_gain, smoothing) in enumerate(grid, start=1):
        controller_config = {
            "radial_gain": radial_gain,
            "tangent_gain": tangent_gain,
            "boundary_gain": boundary_gain,
            "boundary_band": args.boundary_band,
            "boundary_guard": args.boundary_guard,
            "smoothing": smoothing,
        }
        summaries, _ = evaluate_config(
            controller_config=controller_config,
            seed_plan=validation_seed_plan,
            h1_path=args.h1_path,
            h1_model=h1_model,
            max_steps=args.max_steps,
        )
        by_test = {summary.test_name: summary for summary in summaries}
        scripted = by_test["scripted_hand"]
        h1 = by_test["agent_H1"]
        score_mean_tiz = 0.5 * (scripted.tiz_mean + h1.tiz_mean)
        score_min_tiz = min(scripted.tiz_mean, h1.tiz_mean)
        row = {
            "rank_input_order": index,
            **controller_config,
            "score_mean_tiz": score_mean_tiz,
            "score_min_tiz": score_min_tiz,
            **flatten_summary("scripted", scripted),
            **flatten_summary("h1", h1),
        }
        grid_rows.append(row)
        write_csv(grid_csv_path, grid_rows)
        print(
            f"[{index:03d}/{len(grid):03d}] "
            f"radial={radial_gain:.2f} tangent={tangent_gain:.2f} boundary={boundary_gain:.2f} smooth={smoothing:.2f} | "
            f"score={score_mean_tiz:.4f} min={score_min_tiz:.4f} | "
            f"SPC={scripted.tiz_mean:.4f} H1={h1.tiz_mean:.4f}"
        )

    ranked = sorted(
        grid_rows,
        key=lambda row: (row["score_mean_tiz"], row["score_min_tiz"], row["scripted_episode_length_mean"] + row["h1_episode_length_mean"]),
        reverse=True,
    )
    selected = ranked[0]
    selected_config = {
        "radial_gain": selected["radial_gain"],
        "tangent_gain": selected["tangent_gain"],
        "boundary_gain": selected["boundary_gain"],
        "boundary_band": selected["boundary_band"],
        "boundary_guard": selected["boundary_guard"],
        "smoothing": selected["smoothing"],
    }

    print("\nTop 10 validation configurations:")
    for rank, row in enumerate(ranked[:10], start=1):
        print(
            f"#{rank}: score={row['score_mean_tiz']:.4f}, min={row['score_min_tiz']:.4f}, "
            f"SPC={row['scripted_tiz_mean']:.4f}, H1={row['h1_tiz_mean']:.4f}, "
            f"config={{radial={row['radial_gain']}, tangent={row['tangent_gain']}, boundary={row['boundary_gain']}, smoothing={row['smoothing']}}}"
        )

    final_payload = None
    if not args.skip_final:
        print("\nRunning final tuned APF retest...")
        final_seed_plan = spawn_seed_plan(args.seed, args.final_trials, args.final_episodes_per_trial)
        final_summaries, final_trials = evaluate_config(
            controller_config=selected_config,
            seed_plan=final_seed_plan,
            h1_path=args.h1_path,
            h1_model=h1_model,
            max_steps=args.max_steps,
        )
        final_summary_rows = [asdict(summary) for summary in final_summaries]
        for row in final_summary_rows:
            row["robot_name"] = "tuned_reactive_apf"
            row["robot_path"] = "ReactiveAPFController(grid-searched radial+tangent+boundary)"
        final_payload = {
            "schema_version": 1,
            "run_id": run_id,
            "run_dir": str(RUN_DIR),
            "base_seed": args.seed,
            "seed_strategy": "numpy.SeedSequence.spawn shared across test conditions",
            "controller": "Tuned ReactiveAPFController",
            "controller_config": selected_config,
            "selection_metric": "maximize mean validation TIZ over scripted_hand and agent_H1",
            "validation_trials": args.validation_trials,
            "validation_episodes_per_trial": args.validation_episodes_per_trial,
            "final_trials": args.final_trials,
            "final_episodes_per_trial": args.final_episodes_per_trial,
            "max_steps": args.max_steps,
            "results": final_summary_rows,
            "trial_results": final_trials,
        }
        final_json_path = args.out_dir / f"tuned_apf_table2_results_{run_id}.json"
        final_csv_path = args.out_dir / f"tuned_apf_table2_results_{run_id}.csv"
        final_trials_path = args.out_dir / f"tuned_apf_table2_trials_{run_id}.csv"
        final_json_path.write_text(json.dumps(final_payload, indent=2), encoding="utf-8")
        write_csv(final_csv_path, final_summary_rows)
        write_csv(final_trials_path, final_trials)
        for summary in final_summaries:
            print(
                f"  Final {summary.test_name}: TIZ={summary.tiz_mean:.4f} ± {summary.tiz_std:.4f} | "
                f"Len={summary.episode_length_mean:.2f} ± {summary.episode_length_std:.2f} | Catch={summary.catch_rate:.3f}"
            )
        print(f"Saved final JSON: {final_json_path}")
        print(f"Saved final CSV:  {final_csv_path}")
        print(f"Saved final trials: {final_trials_path}")

    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "run_dir": str(RUN_DIR),
        "base_seed": args.seed,
        "h1_path": str(args.h1_path),
        "validation_trials": args.validation_trials,
        "validation_episodes_per_trial": args.validation_episodes_per_trial,
        "max_steps": args.max_steps,
        "selection_metric": "maximize mean validation TIZ over scripted_hand and agent_H1",
        "grid": {
            "radial_gain": radial_values,
            "tangent_gain": tangent_values,
            "boundary_gain": boundary_values,
            "smoothing": smoothing_values,
            "boundary_band": args.boundary_band,
            "boundary_guard": args.boundary_guard,
        },
        "selected_config": selected_config,
        "top_10": ranked[:10],
        "grid_results": grid_rows,
        "final_results": final_payload,
    }
    summary_json_path = args.out_dir / f"apf_grid_search_summary_{run_id}.json"
    summary_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved grid CSV: {grid_csv_path}")
    print(f"Saved grid summary JSON: {summary_json_path}")


if __name__ == "__main__":
    main()
