"""Cross-evaluation script for Robot and Hand agents.

Evaluates pairs of Robot and Hand models and generates a heatmap of TIS scores.
"""

import sys
import os
import json
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO

from src.custom_env import RehabilitationEnv


def calculate_TIS(distance_history, max_steps, z_min, z_max):
    """Calculate Time-Weighted EIR score for a single episode."""
    if len(distance_history) == 0:
        return 0.0
    valid_frames = sum(1 for d in distance_history if z_min <= d <= z_max)
    return valid_frames / max_steps


def evaluate_pair(robot_path, hand_path, num_episodes=50, max_steps=40, z_min=4.0, z_max=6.0):
    """Evaluate a Robot vs Hand pair."""
    hand_name = os.path.basename(os.path.dirname(hand_path)) if hand_path else "scripted_hand"
    print(f"  Evaluating Robot: {os.path.basename(os.path.dirname(robot_path))} vs Hand: {hand_name}")

    robot_model = PPO.load(robot_path, verbose=0)
    hand_model = PPO.load(hand_path, verbose=0) if hand_path else None

    env = RehabilitationEnv(
        training_mode='robot',
        robot_model=robot_model,
        hand_model=hand_model,
        hand_model_paths=None
    )

    env.random_noise = False
    env.max_steps = max_steps

    tis_scores = []
    zpd_coverages = []
    episode_lengths = []
    avg_distances = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        dist_history = []

        while not (done or truncated):
            action, _ = robot_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            dist_history.append(info['dist'])

        zpd_steps = sum(1 for d in dist_history if z_min <= d <= z_max)
        episode_length = len(dist_history)
        tis_scores.append(calculate_TIS(dist_history, max_steps, z_min, z_max))
        zpd_coverages.append(zpd_steps / episode_length if episode_length else 0.0)
        episode_lengths.append(episode_length)
        avg_distances.append(float(np.mean(dist_history)) if dist_history else 0.0)

    env.close()
    return {
        "tis_mean": float(np.mean(tis_scores)),
        "tis_std": float(np.std(tis_scores)),
        "zpd_coverage_mean": float(np.mean(zpd_coverages)),
        "zpd_coverage_std": float(np.std(zpd_coverages)),
        "episode_length_mean": float(np.mean(episode_lengths)),
        "episode_length_std": float(np.std(episode_lengths)),
        "avg_distance_mean": float(np.mean(avg_distances)),
        "avg_distance_std": float(np.std(avg_distances)),
        "num_episodes": int(num_episodes),
        "max_steps": int(max_steps),
    }


def save_results(base_dir, robot_models, hand_models, results, score_matrix, args):
    payload = {
        "base_dir": base_dir,
        "iterations": args.iterations,
        "episodes": args.episodes,
        "max_steps": args.max_steps,
        "zpd_min": args.zpd_min,
        "zpd_max": args.zpd_max,
        "robots": [{"name": name, "path": path} for name, path in robot_models],
        "hands": [{"name": name, "path": path} for name, path in hand_models],
        "score_matrix": score_matrix.tolist(),
        "results": results,
    }

    json_path = os.path.join(base_dir, "cross_eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    csv_path = os.path.join(base_dir, "cross_eval_tis_matrix.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Robot/Hand"] + [name for name, _ in hand_models])
        for i, (r_name, _) in enumerate(robot_models):
            writer.writerow([r_name] + [f"{score_matrix[i, j]:.6f}" for j in range(len(hand_models))])

    print(f"Results saved to: {json_path}")
    print(f"TIS matrix saved to: {csv_path}")


def plot_heatmap(score_matrix, robot_models, hand_models, save_path, vmax=0.5, show=False):
    plt.figure(figsize=(8, 6))

    ax = sns.heatmap(
        score_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=0.0,
        vmax=vmax,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Therapeutic Interaction Score (TIS)'}
    )

    robot_labels = [name for name, _ in robot_models]
    hand_labels = [name for name, _ in hand_models]

    ax.set_xticklabels(hand_labels, rotation=45, ha='right')
    ax.set_yticklabels(robot_labels, rotation=0)

    ax.set_xlabel('Hand Generation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Robot Generation', fontsize=12, fontweight='bold')
    ax.set_title('Robot-Hand Cross-Evaluation Matrix', fontsize=14, pad=15)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Heatmap saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def find_models(base_dir, iterations):
    """Find all robot and hand model paths."""
    robot_models = []
    hand_models = []

    for i in range(1, iterations + 1):
        r_best = os.path.join(base_dir, f"iteration_{i}", "robot", "robot", "best_model.zip")
        r_final = os.path.join(base_dir, f"iteration_{i}", "robot", "robot", "final_model.zip")
        robot_path = r_best if os.path.exists(r_best) else (r_final if os.path.exists(r_final) else None)

        h_best = os.path.join(base_dir, f"iteration_{i}", "hand", "hand", "best_model.zip")
        h_final = os.path.join(base_dir, f"iteration_{i}", "hand", "hand", "final_model.zip")
        hand_path = h_best if os.path.exists(h_best) else (h_final if os.path.exists(h_final) else None)

        if robot_path:
            robot_models.append((f"Robot Gen {i}", robot_path))
        if hand_path:
            hand_models.append((f"Hand Gen {i}", hand_path))

    return robot_models, hand_models


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Cross-evaluate Robot and Hand agents')
    parser.add_argument('--base_dir', type=str,
                        default=r"C:\Users\admin\Desktop\科研\RL\logs\dual_iterative_0415_1041",
                        help='Base log directory containing model iterations')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of iterations to evaluate')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Number of episodes per evaluation')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Max steps per episode')
    parser.add_argument('--zpd_min', type=float, default=4.0,
                        help='ZPD minimum distance')
    parser.add_argument('--zpd_max', type=float, default=6.0,
                        help='ZPD maximum distance')
    parser.add_argument('--vmax', type=float, default=0.5,
                        help='Heatmap color max value')
    parser.add_argument('--show', action='store_true',
                        help='Display the heatmap window after saving')

    args = parser.parse_args()

    robot_models, hand_models = find_models(args.base_dir, args.iterations)

    if not robot_models:
        print("No robot models found!")
        return
    if not hand_models:
        print("No hand models found!")
        return

    print(f"Found {len(robot_models)} robot models and {len(hand_models)} hand models")

    num_robots = len(robot_models)
    num_hands = len(hand_models)
    score_matrix = np.zeros((num_robots, num_hands))

    results = []

    print("\nStarting Cross-Evaluation...")

    for i, (r_name, r_path) in enumerate(robot_models):
        result_row = []
        for j, (h_name, h_path) in enumerate(hand_models):
            metrics = evaluate_pair(
                r_path,
                h_path,
                num_episodes=args.episodes,
                max_steps=args.max_steps,
                z_min=args.zpd_min,
                z_max=args.zpd_max
            )
            metrics.update({
                "robot_index": i + 1,
                "robot_name": r_name,
                "robot_path": r_path,
                "hand_index": j + 1,
                "hand_name": h_name,
                "hand_path": h_path,
            })
            result_row.append(metrics)
            score_matrix[i, j] = metrics["tis_mean"]
            print(f"    --> TIS: {metrics['tis_mean']:.3f} | ZPD: {metrics['zpd_coverage_mean']:.3f} | Len: {metrics['episode_length_mean']:.1f}")
        results.append(result_row)

    print("\nEvaluation Completed! Saving results and generating heatmap...")
    save_results(args.base_dir, robot_models, hand_models, results, score_matrix, args)
    plot_heatmap(
        score_matrix,
        robot_models,
        hand_models,
        os.path.join(args.base_dir, "cross_eval_heatmap.png"),
        vmax=args.vmax,
        show=args.show,
    )


if __name__ == "__main__":
    main()
