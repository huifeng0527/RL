"""Cross-evaluation script for Robot and Hand agents.

Evaluates pairs of Robot and Hand models and generates a heatmap of TIS scores.
"""

import sys
import os

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
    print(f"  Evaluating Robot: {os.path.basename(os.path.dirname(robot_path))} vs Hand: {os.path.basename(os.path.dirname(hand_path))}")

    robot_model = PPO.load(robot_path)
    hand_model = PPO.load(hand_path) if hand_path else None

    env = RehabilitationEnv(
        training_mode='robot',
        robot_model=robot_model,
        hand_model_paths=None
    )

    if hand_model is not None:
        env.hand_model = hand_model

    env.random_noise = False
    env.max_steps = max_steps

    tis_scores = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        dist_history = []

        while not (done or truncated):
            action, _ = robot_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            dist_history.append(info['dist'])

        tis = calculate_TIS(dist_history, max_steps, z_min, z_max)
        tis_scores.append(tis)

    env.close()
    return np.mean(tis_scores)


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
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of iterations to evaluate')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Number of episodes per evaluation')
    parser.add_argument('--max_steps', type=int, default=40,
                        help='Max steps per episode')
    parser.add_argument('--zpd_min', type=float, default=4.0,
                        help='ZPD minimum distance')
    parser.add_argument('--zpd_max', type=float, default=6.0,
                        help='ZPD maximum distance')
    parser.add_argument('--vmax', type=float, default=0.5,
                        help='Heatmap color max value')

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

    print("\nStarting Cross-Evaluation...")

    for i, (r_name, r_path) in enumerate(robot_models):
        for j, (h_name, h_path) in enumerate(hand_models):
            if h_path is None:
                score_matrix[i, j] = np.nan
                print(f"    --> Skipping Hand Gen {j+1} (no model)")
                continue

            score = evaluate_pair(
                r_path, h_path,
                num_episodes=args.episodes,
                max_steps=args.max_steps,
                z_min=args.zpd_min,
                z_max=args.zpd_max
            )
            score_matrix[i, j] = score
            print(f"    --> Score (TIS): {score:.3f}")

    print("\nEvaluation Completed! Generating Heatmap...")

    plt.figure(figsize=(8, 6))

    ax = sns.heatmap(
        score_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=0.0,
        vmax=args.vmax,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Therapeutic Interaction Score (TIS)'}
    )

    robot_labels = [name for name, _ in robot_models]
    hand_labels = [name for name, _ in hand_models]

    ax.set_xticklabels(hand_labels, rotation=45, ha='right')
    ax.set_yticklabels(robot_labels, rotation=0)

    ax.set_xlabel('Hand', fontsize=12, fontweight='bold')
    ax.set_ylabel('Robot', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Evaluation Matrix', fontsize=14, pad=15)
    ax.invert_yaxis()

    plt.tight_layout()

    save_path = os.path.join(args.base_dir, "cross_eval_heatmap.png")
    plt.savefig(save_path, dpi=300)
    print(f"Heatmap saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    main()
