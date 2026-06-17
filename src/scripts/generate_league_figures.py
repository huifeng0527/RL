"""Generate league-training analysis figures.

Uses cross_eval_results.json produced by cross_eval.py to reconstruct PFSP
sampling priorities and to compare league training against baseline robots.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib.pyplot as plt
import numpy as np

from src.scripts.cross_eval import evaluate_pair, find_models


def load_cross_eval(base_dir):
    path = os.path.join(base_dir, "cross_eval_results.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_pfsp_probs(avg_lengths, min_prob_total=0.05):
    scores = np.array([1.0 / max(float(length), 1.0) for length in avg_lengths], dtype=float)
    raw_probs = scores / scores.sum() if scores.sum() > 0 else np.ones(len(scores)) / len(scores)
    min_prob = min_prob_total / len(scores)
    probs = np.maximum(raw_probs, min_prob)
    return probs / probs.sum()


def generate_pfsp_distribution(data, base_dir, generations=(3, 5, 7, 10)):
    pfsp = {}
    fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharey=True)
    axes = axes.ravel()

    for ax, gen in zip(axes, generations):
        row = data["results"][gen - 1]
        available = row[:gen]
        lengths = [cell["episode_length_mean"] for cell in available]
        probs = compute_pfsp_probs(lengths)
        labels = [f"H{i}" for i in range(1, gen + 1)]

        pfsp[f"robot_gen_{gen}"] = {
            "hand_generations": list(range(1, gen + 1)),
            "episode_length_mean": lengths,
            "selection_probability": probs.tolist(),
        }

        ax.bar(labels, probs, color="#4C72B0")
        ax.axhline(0.05 / gen, color="#C44E52", linestyle="--", linewidth=1, label="floor")
        ax.set_title(f"Robot Gen {gen}")
        ax.set_ylim(0, max(0.5, float(np.max(probs)) * 1.2))
        ax.set_ylabel("Sampling Probability")
        ax.tick_params(axis="x", rotation=45)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Post-hoc PFSP Sampling Priorities")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    json_path = os.path.join(base_dir, "pfsp_distribution.json")
    png_path = os.path.join(base_dir, "pfsp_distribution.png")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pfsp, f, indent=2)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"PFSP distribution saved to: {json_path}")
    print(f"PFSP figure saved to: {png_path}")


def evaluate_robot_against_pool(robot_name, robot_path, hand_models, episodes, max_steps, zpd_min, zpd_max):
    per_hand = []
    for hand_index, (hand_name, hand_path) in enumerate(hand_models, start=1):
        metrics = evaluate_pair(
            robot_path,
            hand_path,
            num_episodes=episodes,
            max_steps=max_steps,
            z_min=zpd_min,
            z_max=zpd_max,
        )
        metrics.update({
            "robot_name": robot_name,
            "robot_path": robot_path,
            "hand_index": hand_index,
            "hand_name": hand_name,
            "hand_path": hand_path,
        })
        per_hand.append(metrics)
    return per_hand


def summarize_pool_metrics(per_hand):
    keys = ["tis_mean", "zpd_coverage_mean", "episode_length_mean"]
    summary = {}
    for key in keys:
        values = np.array([item[key] for item in per_hand], dtype=float)
        summary[key] = float(values.mean())
        summary[f"{key}_std_across_hands"] = float(values.std())
    return summary


def generate_baseline_comparison(base_dir, baseline_a, baseline_b, episodes, max_steps, zpd_min, zpd_max):
    robot_models, hand_models = find_models(base_dir, 10)
    if len(robot_models) < 10 or len(hand_models) < 10:
        raise RuntimeError("Expected 10 robot and 10 hand models in the league directory.")

    final_league_robot = robot_models[-1]
    comparisons = [("PFSP League", final_league_robot[1])]
    if baseline_a:
        comparisons.append(("Scripted Only", baseline_a))
    if baseline_b:
        comparisons.append(("Single RL Hand", baseline_b))

    results = {}
    summaries = {}
    for robot_name, robot_path in comparisons:
        print(f"\nEvaluating baseline: {robot_name}")
        per_hand = evaluate_robot_against_pool(
            robot_name,
            robot_path,
            hand_models,
            episodes,
            max_steps,
            zpd_min,
            zpd_max,
        )
        results[robot_name] = per_hand
        summaries[robot_name] = summarize_pool_metrics(per_hand)

    payload = {
        "episodes": episodes,
        "max_steps": max_steps,
        "zpd_min": zpd_min,
        "zpd_max": zpd_max,
        "summaries": summaries,
        "results": results,
    }
    json_path = os.path.join(base_dir, "baseline_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    metric_keys = ["tis_mean", "zpd_coverage_mean", "episode_length_mean"]
    metric_labels = ["TIS", "ZPD Coverage", "Episode Length"]
    names = list(summaries.keys())
    x = np.arange(len(metric_keys))
    width = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(7, 4))
    for idx, name in enumerate(names):
        values = [summaries[name][key] for key in metric_keys]
        errors = [summaries[name][f"{key}_std_across_hands"] for key in metric_keys]
        ax.bar(x + idx * width - 0.4 + width / 2, values, width, yerr=errors, capsize=3, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Mean across 10 hand generations")
    ax.set_title("League Training vs. Baselines")
    ax.legend()
    fig.tight_layout()

    png_path = os.path.join(base_dir, "baseline_comparison.png")
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Baseline comparison saved to: {json_path}")
    print(f"Baseline figure saved to: {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate league training figures")
    parser.add_argument("--base_dir", default="src/logs/dual_iterative_0509_0945")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max_steps", type=int, default=40)
    parser.add_argument("--zpd_min", type=float, default=4.0)
    parser.add_argument("--zpd_max", type=float, default=6.0)
    parser.add_argument("--baseline_a", type=str, default=None, help="Path to scripted-only baseline robot")
    parser.add_argument("--baseline_b", type=str, default=None, help="Path to single-RL-hand baseline robot")
    parser.add_argument("--skip_baseline", action="store_true")
    args = parser.parse_args()

    data = load_cross_eval(args.base_dir)
    generate_pfsp_distribution(data, args.base_dir)

    if not args.skip_baseline:
        generate_baseline_comparison(
            args.base_dir,
            args.baseline_a,
            args.baseline_b,
            args.episodes,
            args.max_steps,
            args.zpd_min,
            args.zpd_max,
        )


if __name__ == "__main__":
    main()
