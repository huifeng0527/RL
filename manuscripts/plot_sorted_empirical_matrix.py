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
from src.observation_schema import INTERACTION_HISTORY_CHANNELS, model_obs_dim, obs_dim

RUN_DIR = Path("logs/league_relaxed_hand_reward_10iter_h1m")
OUT = Path("manuscripts/league_relaxed_hand_reward_sorted_matrix")
OUT.mkdir(exist_ok=True)
EPISODES = 30
MAX_STEPS = 100
HISTORY_LENGTH = 16

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def model_path(iteration, agent):
    base = RUN_DIR / f"iteration_{iteration}" / agent / agent
    best = base / "best_model.zip"
    final = base / "final_model.zip"
    if best.exists():
        return best
    if final.exists():
        return final
    return None


def complete_generations():
    gens = []
    for i in range(1, 51):
        if model_path(i, "robot") is not None and model_path(i, "hand") is not None:
            gens.append(i)
    return gens


def infer_history_mode(robot_model):
    expected_dim = model_obs_dim(robot_model)
    interaction_dim = obs_dim(HISTORY_LENGTH, 0, INTERACTION_HISTORY_CHANNELS)
    return "interaction" if expected_dim == interaction_dim else "motion"


def evaluate_pair(robot_path, hand_path):
    robot_model = PPO.load(str(robot_path), verbose=0)
    hand_model = PPO.load(str(hand_path), custom_objects={"learning_rate": 0.0, "optimizer_class": None}, verbose=0)
    env = RehabilitationEnv(
        training_mode="robot",
        hand_model=hand_model,
        history_length=HISTORY_LENGTH,
        history_mode=infer_history_mode(robot_model),
    )
    env.random_noise = False
    env.max_steps = MAX_STEPS

    tis_values = []
    zpd_values = []
    too_close_values = []
    too_far_values = []
    length_values = []
    z_min = float(env.zpd_min)
    z_max = float(env.zpd_max)

    for _ in range(EPISODES):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        distances = []
        while not (terminated or truncated):
            action, _ = robot_model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            distances.append(float(info["dist"]))
        if not distances:
            continue
        distances = np.asarray(distances, dtype=float)
        in_zpd = (distances >= z_min) & (distances <= z_max)
        tis_values.append(float(np.sum(in_zpd) / MAX_STEPS))
        zpd_values.append(float(np.mean(in_zpd)))
        too_close_values.append(float(np.mean(distances < z_min)))
        too_far_values.append(float(np.mean(distances > z_max)))
        length_values.append(float(len(distances)))

    env.close()
    return {
        "tis_mean": float(np.mean(tis_values)),
        "zpd_coverage_mean": float(np.mean(zpd_values)),
        "too_close_rate_mean": float(np.mean(too_close_values)),
        "too_far_rate_mean": float(np.mean(too_far_values)),
        "episode_length_mean": float(np.mean(length_values)),
    }


def build_cross_eval(gens):
    cache = OUT / "cross_eval_relaxed_h1m_7gen.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    tis = np.zeros((len(gens), len(gens)), dtype=float)
    results = []
    for i, r_gen in enumerate(gens):
        row = []
        for j, h_gen in enumerate(gens):
            print(f"Evaluating R{r_gen} vs H{h_gen}")
            metrics = evaluate_pair(model_path(r_gen, "robot"), model_path(h_gen, "hand"))
            metrics.update({"robot_generation": r_gen, "hand_generation": h_gen})
            row.append(metrics)
            tis[i, j] = metrics["tis_mean"]
            print(f"  TIS={metrics['tis_mean']:.3f}, ZPD={metrics['zpd_coverage_mean']:.3f}")
        results.append(row)

    payload = {
        "run_dir": str(RUN_DIR),
        "generations": gens,
        "episodes": EPISODES,
        "max_steps": MAX_STEPS,
        "zpd_min": 3.5,
        "zpd_max": 5.5,
        "tis_matrix": tis.tolist(),
        "results": results,
    }
    cache.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (OUT / "cross_eval_relaxed_h1m_7gen_tis.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Robot/Hand"] + [f"H{i}" for i in gens])
        for i, r_gen in enumerate(gens):
            writer.writerow([f"R{r_gen}"] + [f"{tis[i, j]:.6f}" for j in range(len(gens))])
    return payload


def polish_axis(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3)


def plot_matrix(ax, matrix, row_labels, col_labels, title):
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="YlGnBu",
        vmin=0,
        vmax=max(0.8, float(np.nanmax(matrix)) + 0.05),
        annot=True,
        fmt=".2f",
        linewidths=0.35,
        cbar=True,
        cbar_kws={"label": "TIS", "shrink": 0.82, "pad": 0.01},
        annot_kws={"fontsize": 7},
    )
    ax.set_xticklabels(col_labels, rotation=0)
    ax.set_yticklabels(row_labels, rotation=0)
    ax.set_xlabel("Hand")
    ax.set_ylabel("Robot")
    ax.set_title(title)


def main():
    gens = complete_generations()
    if not gens:
        raise RuntimeError(f"No complete robot/hand generations found in {RUN_DIR}")
    cross = build_cross_eval(gens)
    tis = np.asarray(cross["tis_matrix"], dtype=float)

    robot_strength = tis.mean(axis=1)
    hand_easiness = tis.mean(axis=0)
    robot_order = np.argsort(robot_strength)
    hand_order = np.argsort(-hand_easiness)
    sorted_tis = tis[np.ix_(robot_order, hand_order)]

    sorted_robot_labels = [f"R{gens[i]}\n{robot_strength[i]:.2f}" for i in robot_order]
    sorted_hand_labels = [f"H{gens[j]}\n{hand_easiness[j]:.2f}" for j in hand_order]
    generation_labels_r = [f"R{i}" for i in gens]
    generation_labels_h = [f"H{i}" for i in gens]

    summary = {
        "robot_strength": {f"R{gens[i]}": float(robot_strength[i]) for i in range(len(gens))},
        "hand_easiness": {f"H{gens[j]}": float(hand_easiness[j]) for j in range(len(gens))},
        "robot_order_weak_to_strong": [f"R{gens[i]}" for i in robot_order],
        "hand_order_easy_to_hard": [f"H{gens[j]}" for j in hand_order],
    }
    (OUT / "sorted_strength_difficulty_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), dpi=300, constrained_layout=True)
    plot_matrix(axes[0], tis, generation_labels_r, generation_labels_h, "Generation order")
    plot_matrix(axes[1], sorted_tis, sorted_robot_labels, sorted_hand_labels, "Sorted by robot strength and hand difficulty")
    axes[1].set_xlabel("Hand, easy → hard")
    axes[1].set_ylabel("Robot, weak → strong")
    for ax in axes:
        polish_axis(ax)
    fig.savefig(OUT / "empirical_tis_matrix_sorted_strength_difficulty.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=300, constrained_layout=True)
    plot_matrix(ax, sorted_tis, sorted_robot_labels, sorted_hand_labels, "Empirical TIS matrix\n(sorted strength/difficulty)")
    ax.set_xlabel("Hand, easy → hard")
    ax.set_ylabel("Robot, weak → strong")
    polish_axis(ax)
    fig.savefig(OUT / "empirical_tis_matrix_sorted_only.png", bbox_inches="tight")
    plt.close(fig)

    print((OUT / "empirical_tis_matrix_sorted_strength_difficulty.png").as_posix())
    print((OUT / "empirical_tis_matrix_sorted_only.png").as_posix())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
