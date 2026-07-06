from pathlib import Path
import csv
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from stable_baselines3 import PPO

from src.custom_env import RehabilitationEnv

BASE_DIR = Path("logs/league_paper_gru_multistep_aux_pfsp_window_20iter")
OUT_DIR = Path("manuscripts/current_league_pfsp_window_7iter")
OUT_DIR.mkdir(exist_ok=True)
ITERATIONS = 7
EPISODES = 50
MAX_STEPS = 100
ZPD_MIN = 4.0
ZPD_MAX = 6.0


def model_path(iteration, agent):
    base = BASE_DIR / f"iteration_{iteration}" / agent / agent
    best = base / "best_model.zip"
    final = base / "final_model.zip"
    return best if best.exists() else final


def evaluate_pair(robot_path, hand_path):
    robot_model = PPO.load(str(robot_path), verbose=0)
    hand_model = PPO.load(str(hand_path), verbose=0)
    env = RehabilitationEnv(
        training_mode="robot",
        robot_model=robot_model,
        hand_model=hand_model,
        history_length=16,
        history_mode="interaction",
    )
    env.random_noise = False
    env.max_steps = MAX_STEPS

    values = {"tis": [], "zpd": [], "length": [], "too_close": [], "too_far": []}
    done_counts = {}
    for _ in range(EPISODES):
        obs, _ = env.reset()
        done = False
        truncated = False
        distances = []
        last_info = {}
        while not (done or truncated):
            action, _ = robot_model.predict(obs, deterministic=True)
            obs, _, done, truncated, info = env.step(action)
            distances.append(float(info["dist"]))
            last_info = info
        episode_length = len(distances)
        if episode_length:
            zpd_steps = sum(ZPD_MIN <= d <= ZPD_MAX for d in distances)
            too_close = sum(d < ZPD_MIN for d in distances) / episode_length
            too_far = sum(d > ZPD_MAX for d in distances) / episode_length
            values["tis"].append(zpd_steps / MAX_STEPS)
            values["zpd"].append(zpd_steps / episode_length)
            values["length"].append(episode_length)
            values["too_close"].append(too_close)
            values["too_far"].append(too_far)
        reason = last_info.get("done_reason", "unknown")
        done_counts[reason] = done_counts.get(reason, 0) + 1
    env.close()
    return {
        "tis_mean": float(np.mean(values["tis"])),
        "tis_std": float(np.std(values["tis"])),
        "zpd_coverage_mean": float(np.mean(values["zpd"])),
        "episode_length_mean": float(np.mean(values["length"])),
        "too_close_rate_mean": float(np.mean(values["too_close"])),
        "too_far_rate_mean": float(np.mean(values["too_far"])),
        "done_reason_counts": done_counts,
    }


robots = [(f"R{i}", model_path(i, "robot")) for i in range(1, ITERATIONS + 1)]
hands = [(f"H{i}", model_path(i, "hand")) for i in range(1, ITERATIONS + 1)]
results = []
tis_matrix = np.zeros((ITERATIONS, ITERATIONS), dtype=float)
zpd_matrix = np.zeros((ITERATIONS, ITERATIONS), dtype=float)

for i, (robot_name, robot_zip) in enumerate(robots):
    for j, (hand_name, hand_zip) in enumerate(hands):
        print(f"Evaluating {robot_name} vs {hand_name}")
        metrics = evaluate_pair(robot_zip, hand_zip)
        metrics.update({
            "robot": robot_name,
            "hand": hand_name,
            "robot_path": str(robot_zip),
            "hand_path": str(hand_zip),
            "episodes": EPISODES,
            "max_steps": MAX_STEPS,
        })
        results.append(metrics)
        tis_matrix[i, j] = metrics["tis_mean"]
        zpd_matrix[i, j] = metrics["zpd_coverage_mean"]
        print(f"  TIS={metrics['tis_mean']:.3f}, ZPD={metrics['zpd_coverage_mean']:.3f}")

payload = {
    "base_dir": str(BASE_DIR),
    "iterations": ITERATIONS,
    "episodes": EPISODES,
    "max_steps": MAX_STEPS,
    "zpd_min": ZPD_MIN,
    "zpd_max": ZPD_MAX,
    "results": results,
    "tis_matrix": tis_matrix.tolist(),
    "zpd_matrix": zpd_matrix.tolist(),
}
with (OUT_DIR / "cross_iter_validation_7iter.json").open("w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

with (OUT_DIR / "cross_iter_validation_tis_7iter.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Robot/Hand"] + [name for name, _ in hands])
    for i, (name, _) in enumerate(robots):
        writer.writerow([name] + [f"{tis_matrix[i, j]:.6f}" for j in range(ITERATIONS)])

fig, ax = plt.subplots(figsize=(6.4, 5.4), dpi=300)
sns.heatmap(
    tis_matrix,
    ax=ax,
    cmap="YlGnBu",
    vmin=0,
    vmax=max(0.8, float(np.nanmax(tis_matrix))),
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.5,
    cbar_kws={"label": "Mean TIS"},
)
ax.set_xticklabels([name for name, _ in hands], rotation=0)
ax.set_yticklabels([name for name, _ in robots], rotation=0)
ax.set_xlabel("Hand generation")
ax.set_ylabel("Robot generation")
ax.set_title("Cross-iteration validation: robot vs. hand generations")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_cross_iter_validation_tis_heatmap_7iter.png", bbox_inches="tight")
plt.close(fig)

row_mean = tis_matrix.mean(axis=1)
row_worst = tis_matrix.min(axis=1)
fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=300)
xs = np.arange(1, ITERATIONS + 1)
ax.plot(xs, row_mean, "-o", lw=2, label="Mean across hands", color="#1f77b4")
ax.plot(xs, row_worst, "-s", lw=2, label="Worst hand", color="#d62728")
ax.set_xlabel("Robot generation")
ax.set_ylabel("TIS")
ax.set_xticks(xs)
ax.set_ylim(0, max(0.8, float(row_mean.max()) + 0.08))
ax.grid(alpha=0.25)
ax.legend(frameon=False)
ax.set_title("Cross-generation robustness summary")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig_cross_iter_validation_summary_7iter.png", bbox_inches="tight")
plt.close(fig)

print(f"Saved cross-iteration validation outputs to {OUT_DIR}")
