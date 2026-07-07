from pathlib import Path
import json
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from src.custom_env import RehabilitationEnv
from src.observation_schema import HISTORY_CHANNELS, INTERACTION_HISTORY_CHANNELS, history_slice
try:
    import generate_pfsp_window_aux_visual as aux
except ModuleNotFoundError:
    import generate_aux_prediction_visual as aux


def _relative_paths(ex):
    current_hand = np.asarray(ex["current_hand_position"], dtype=float)
    past_hand = np.asarray(ex["past_hand_positions"], dtype=float) - current_hand
    past_robot = np.asarray(ex["past_robot_positions"], dtype=float) - current_hand
    current_robot = np.asarray(ex["current_robot_position"], dtype=float) - current_hand
    future_hand = np.asarray(ex["future_hand_positions"], dtype=float) - current_hand
    future_robot = np.asarray(ex["future_robot_positions"], dtype=float) - current_hand
    pred_future = np.cumsum(np.asarray(ex["pred_moves"], dtype=float), axis=0)
    return past_hand, past_robot, current_robot, np.zeros(2, dtype=float), future_robot, pred_future, future_hand


if not hasattr(aux, "relative_paths"):
    aux.relative_paths = _relative_paths

if not Path(aux.ROBOT_PATH).exists():
    aux_run_dir = Path("logs/league_paper_gru_multistep_aux_pfsp_window_20iter")
    aux.ROBOT_PATH = aux_run_dir / "iteration_9" / "robot" / "robot" / "final_model.zip"
    aux.HAND_PATH = aux_run_dir / "iteration_8" / "hand" / "hand" / "final_model.zip"

ABLATION_DIR = Path("logs/ablation_gru_h1_h10_0626_2136")
ABLATION_EXCEL = ABLATION_DIR / "ablation_review_raw.xlsx"
SMOOTH_WINDOW = 200000
MAX_PLOT_POINTS = 1500
OUT_DIR = Path("manuscripts/current_ablation_gru_h1_h10_final")
OUT_DIR.mkdir(exist_ok=True)
OUT_FIG = OUT_DIR / "paper_fig_network_ablation_aux_composite.png"
OUT_FIG_NOTITLE = OUT_DIR / "paper_fig_network_ablation_aux_composite_no_title.png"
OUT_FIG_FILLED = OUT_DIR / "paper_fig_network_ablation_aux_composite_filled.png"
OUT_FIG_FILLED_NOTITLE = OUT_DIR / "paper_fig_network_ablation_aux_composite_filled_no_title.png"

METHODS = [
    ("MLP", "1_MLP", "#4C72B0"),
    ("GRU", "2_GRU_Seq", "#DD8452"),
    ("Auxiliary GRU", "3_GRU_Aux", "#55A868"),
]

FINAL_QUALITY_JSON = OUT_DIR / "ablation_final_task_quality_metrics.json"
FINAL_QUALITY_EPISODES = 60
MAX_WORKSPACE_AREA = (15.0 - 2 * 0.6) * (10.0 - 2 * 0.6)

plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def polish_axis(ax):
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
    ax.tick_params(width=0.8, length=3)


def read_raw_episode_data():
    xls = pd.ExcelFile(ABLATION_EXCEL)
    raw_sheets = [s for s in xls.sheet_names if s.startswith("Episode_Raw")]
    frames = [pd.read_excel(xls, sheet_name=sheet) for sheet in raw_sheets]
    data = pd.concat(frames, ignore_index=True)
    return data.sort_values(["group", "timestep", "episode_index"])


def bin_by_timestep(sub, metric, max_points=MAX_PLOT_POINTS):
    x = sub["timestep"].to_numpy(dtype=float)
    y = sub[metric].to_numpy(dtype=float)
    if len(x) <= max_points:
        return x, y

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    bin_edges = np.linspace(x_min, x_max, max_points + 1)
    bin_idx = np.clip(np.searchsorted(bin_edges, x, side="right") - 1, 0, max_points - 1)
    sums = np.bincount(bin_idx, weights=y, minlength=max_points)
    counts = np.bincount(bin_idx, minlength=max_points)
    valid = counts > 0
    x_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return x_centers[valid], sums[valid] / counts[valid]


def smooth_by_timesteps(x, y, smooth_window=SMOOTH_WINDOW):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if smooth_window <= 1 or len(y) < 3:
        return x, y

    bin_width = float(np.median(np.diff(x))) if len(x) > 1 else smooth_window
    rolling_points = max(1, int(round(float(smooth_window) / max(bin_width, 1.0))))
    if rolling_points <= 1:
        return x, y
    y_smooth = pd.Series(y).rolling(window=rolling_points, center=True, min_periods=max(1, rolling_points // 4)).mean().to_numpy()
    valid = ~np.isnan(y_smooth)
    return x[valid], y_smooth[valid]


def metric_curve(raw_df, group, metric):
    sub = raw_df[raw_df["group"] == group].sort_values("timestep")
    x, y = bin_by_timestep(sub, metric)
    return smooth_by_timesteps(x, y)


def load_metrics():
    raw_df = read_raw_episode_data()
    loaded = {}
    for label, group, color in METHODS:
        reward_steps, reward = metric_curve(raw_df, group, "reward")
        length_steps, length = metric_curve(raw_df, group, "episode_length")
        loaded[label] = {
            "reward_steps": reward_steps / 1_000_000,
            "reward": reward,
            "length_steps": length_steps / 1_000_000,
            "length": length,
            "color": color,
        }
    return loaded


def plot_reward(ax, metrics):
    for label, data in metrics.items():
        ax.plot(data["reward_steps"], data["reward"], lw=2.0, label=label, color=data["color"])
    ax.set_xlabel("Training steps (M)")
    ax.set_ylabel("Episode reward")
    ax.set_title("(a) Reward during GRU ablation")
    polish_axis(ax)
    ax.legend(frameon=False, loc="lower right")


def plot_length(ax, metrics):
    for label, data in metrics.items():
        ax.plot(data["length_steps"], data["length"], lw=2.0, label=label, color=data["color"])
    ax.set_xlabel("Training steps (M)")
    ax.set_ylabel("Episode length")
    ax.set_title("(b) Episode length during GRU ablation")
    polish_axis(ax)


def choose_aux_examples():
    robot_model = PPO.load(str(aux.ROBOT_PATH), verbose=0)
    hand_model = PPO.load(str(aux.HAND_PATH), custom_objects={"learning_rate": 0.0, "optimizer_class": None}, verbose=0)

    env = RehabilitationEnv(
        training_mode="robot",
        hand_model=hand_model,
        history_length=aux.HISTORY_LENGTH,
        history_mode="interaction",
    )
    env.random_noise = False
    env.max_steps = 100

    h_slice = history_slice(aux.HISTORY_LENGTH, INTERACTION_HISTORY_CHANNELS)
    latest_step_start = h_slice.stop - INTERACTION_HISTORY_CHANNELS
    hand_delta_start = latest_step_start + INTERACTION_HISTORY_CHANNELS - HISTORY_CHANNELS

    candidates = []
    horizon_errors = []
    for _ in range(140):
        obs, _ = env.reset()
        obs_seq, hand_pos_seq, robot_pos_seq = [], [], []
        done = False
        truncated = False
        while not (done or truncated):
            obs_seq.append(np.asarray(obs, dtype=np.float32).copy())
            hand_pos_seq.append(np.asarray(env.hand_position, dtype=np.float32).copy())
            robot_pos_seq.append(np.asarray(env.robot_position, dtype=np.float32).copy())
            action, _ = robot_model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)

        if len(obs_seq) <= aux.HISTORY_LENGTH + aux.FUTURE_HORIZON + 2:
            continue
        obs_arr = np.asarray(obs_seq, dtype=np.float32)
        hand_pos_arr = np.asarray(hand_pos_seq, dtype=np.float32)
        robot_pos_arr = np.asarray(robot_pos_seq, dtype=np.float32)

        for t in range(aux.HISTORY_LENGTH, len(obs_arr) - aux.FUTURE_HORIZON):
            pred_moves, pred_risk = aux.get_future_prediction(robot_model, obs_arr[t])
            future_obs = obs_arr[t + 1:t + 1 + aux.FUTURE_HORIZON]
            true_moves = future_obs[:, hand_delta_start:h_slice.stop]
            pred_traj = np.cumsum(pred_moves, axis=0)
            true_traj = np.cumsum(true_moves, axis=0)
            traj_errors = np.mean((pred_traj - true_traj) ** 2, axis=1)
            horizon_errors.append(traj_errors)
            total_true = true_traj[-1]
            total_pred = pred_traj[-1]
            cosine = float(np.dot(total_true, total_pred) / ((np.linalg.norm(total_true) * np.linalg.norm(total_pred)) + 1e-8))
            true_step_lengths = np.linalg.norm(true_moves, axis=1)
            candidates.append({
                "t": int(t),
                "pred_moves": pred_moves,
                "true_moves": true_moves,
                "past_hand_positions": hand_pos_arr[max(0, t - 7):t + 1],
                "past_robot_positions": robot_pos_arr[max(0, t - 7):t + 1],
                "current_hand_position": hand_pos_arr[t],
                "current_robot_position": robot_pos_arr[t],
                "future_hand_positions": hand_pos_arr[t + 1:t + 1 + aux.FUTURE_HORIZON],
                "future_robot_positions": robot_pos_arr[t + 1:t + 1 + aux.FUTURE_HORIZON],
                "pred_risk": pred_risk,
                "trajectory_mse": float(np.mean(traj_errors)),
                "direction_cosine": cosine,
                "true_step_mean": float(np.mean(true_step_lengths)),
                "true_step_std": float(np.std(true_step_lengths)),
            })
    env.close()

    if not candidates:
        raise RuntimeError("No auxiliary prediction candidates collected.")

    step_means = np.array([c["true_step_mean"] for c in candidates], dtype=float)
    target_step = float(np.median(step_means))
    tolerance = max(0.03, 0.18 * target_step)
    uniform_pool = [c for c in candidates if abs(c["true_step_mean"] - target_step) <= tolerance]
    if len(uniform_pool) < 40:
        tolerance = max(0.05, 0.30 * target_step)
        uniform_pool = [c for c in candidates if abs(c["true_step_mean"] - target_step) <= tolerance]
    if len(uniform_pool) < 10:
        uniform_pool = candidates

    ranked = sorted(uniform_pool, key=lambda x: x["trajectory_mse"])
    low1 = ranked[max(0, int(0.05 * (len(ranked) - 1)))]
    low2 = ranked[max(0, int(0.12 * (len(ranked) - 1)))]
    similar_pool = [c for c in ranked if c["direction_cosine"] > 0.65]
    if similar_pool:
        similar_pool = sorted(similar_pool, key=lambda x: abs(x["trajectory_mse"] - np.percentile([v["trajectory_mse"] for v in ranked], 55)))
        medium = similar_pool[0]
    else:
        medium = ranked[int(0.55 * (len(ranked) - 1))]
    high = ranked[int(0.92 * (len(ranked) - 1))]

    selected = []
    seen = set()
    for item in [low1, low2, medium, high]:
        key = (item["t"], round(item["trajectory_mse"], 6))
        if key not in seen:
            seen.add(key)
            selected.append(item)
    while len(selected) < 4:
        item = ranked[int((0.2 + 0.2 * len(selected)) * (len(ranked) - 1))]
        key = (item["t"], round(item["trajectory_mse"], 6))
        if key not in seen:
            seen.add(key)
            selected.append(item)
    return selected[:4], np.asarray(horizon_errors, dtype=np.float32)


def plot_aux_examples(subspec, fig, examples):
    inner = subspec.subgridspec(2, 2, hspace=0.28, wspace=0.18)
    axes = [fig.add_subplot(inner[i, j]) for i in range(2) for j in range(2)]
    selected = examples[:4]
    for idx, (ax, ex) in enumerate(zip(axes, selected), start=1):
        past_hand, past_robot, current_robot, _, _, pred_future, true_future = aux.relative_paths(ex)
        past_hand = past_hand[-8:]
        past_robot = past_robot[-8:]
        ax.plot(past_hand[:, 0], past_hand[:, 1], "o-", color="#8c8c8c", lw=1.0, ms=2.1, label="hand past 8")
        ax.plot(past_robot[:, 0], past_robot[:, 1], ".-", color="#d62728", lw=0.9, ms=2.0, alpha=0.7, label="robot past 8")
        ax.scatter([0], [0], marker="*", s=48, color="black", zorder=6, label="hand now")
        ax.scatter([current_robot[0]], [current_robot[1]], marker="X", s=44, color="#d62728", zorder=6, label="robot now")
        ax.plot(true_future[:, 0], true_future[:, 1], "o-", color="#1f77b4", lw=1.45, ms=2.4, label="actual future")
        ax.plot(pred_future[:, 0], pred_future[:, 1], "s--", color="#ff7f0e", lw=1.45, ms=2.4, label="predicted future")
        ax.text(0.03, 0.94, f"MSE={ex['trajectory_mse']:.2f}\nstep={ex['true_step_mean']:.2f}", transform=ax.transAxes, fontsize=6.6, va="top")
        ax.set_xlabel("rel. x", fontsize=7)
        ax.set_ylabel("rel. y", fontsize=7)
        ax.tick_params(labelsize=6.5, width=0.7, length=2.5)
        polish_axis(ax)
        ax.set_aspect("auto")
    axes[0].text(-0.12, 1.20, "(c) Auxiliary future prediction examples", transform=axes[0].transAxes, fontsize=10, fontweight="normal")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, frameon=False, fontsize=5.8, loc="best")


def collect_final_quality_metrics():
    if FINAL_QUALITY_JSON.exists():
        with open(FINAL_QUALITY_JSON, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(ABLATION_DIR / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    hand_paths = config.get("hand_paths", [])
    scripted_prob = float(config.get("scripted_hand_sample_prob", 0.1))
    history_length = int(config.get("history_length", 16))
    history_mode = str(config.get("history_mode", "interaction"))

    results = {}
    for method_idx, (label, group, color) in enumerate(METHODS):
        random.seed(2026)
        np.random.seed(2026)
        model_path = ABLATION_DIR / group / "final_model.zip"
        model = PPO.load(
            str(model_path),
            custom_objects={"learning_rate": 0.0, "optimizer_class": None},
            verbose=0,
        )
        env = RehabilitationEnv(
            training_mode="robot",
            hand_model_paths=hand_paths,
            history_length=history_length,
            history_mode=history_mode,
        )
        env.scripted_hand_sample_prob = scripted_prob
        env.random_noise = False
        z_min = float(env.zpd_min)
        z_max = float(env.zpd_max)
        max_steps = int(env.max_steps)

        tis_values = []
        workspace_values = []
        jerk_values = []
        for _ in range(FINAL_QUALITY_EPISODES):
            obs, _ = env.reset()
            terminated = False
            truncated = False
            in_zpd = 0
            xs = [float(env.robot_position[0])]
            ys = [float(env.robot_position[1])]
            terminal_info = None
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                dist = float(info.get("dist", 0.0))
                if z_min <= dist <= z_max:
                    in_zpd += 1
                robot_pos = info.get("robot_pos")
                if robot_pos is not None:
                    xs.append(float(robot_pos[0]))
                    ys.append(float(robot_pos[1]))
                terminal_info = info

            tis_values.append(in_zpd / max(max_steps, 1))
            if len(xs) > 1:
                area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                workspace_values.append(area / MAX_WORKSPACE_AREA)
            else:
                workspace_values.append(0.0)
            league_episode = (terminal_info or {}).get("league_episode", {})
            jerk_values.append(float(league_episode.get("mean_jerk", np.mean(env._episode_jerks) if env._episode_jerks else 0.0)))

        env.close()
        results[label] = {
            "color": color,
            "episodes": FINAL_QUALITY_EPISODES,
            "tis_mean": float(np.mean(tis_values)),
            "tis_std": float(np.std(tis_values, ddof=0)),
            "workspace_mean": float(np.mean(workspace_values)),
            "workspace_std": float(np.std(workspace_values, ddof=0)),
            "mean_jerk_mean": float(np.mean(jerk_values)),
            "mean_jerk_std": float(np.std(jerk_values, ddof=0)),
        }

    with open(FINAL_QUALITY_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results


def plot_final_quality(ax, quality):
    metric_specs = [
        ("tis_mean", "TIS"),
        ("workspace_mean", "Workspace\ncoverage"),
        ("mean_jerk_mean", "Mean\njerk ↓"),
    ]
    baseline = quality["MLP"]
    x = np.arange(len(metric_specs))
    width = 0.24
    offsets = np.linspace(-width, width, len(METHODS))
    for offset, (label, _, color) in zip(offsets, METHODS):
        values = []
        for key, _ in metric_specs:
            denom = max(float(baseline[key]), 1e-8)
            values.append(float(quality[label][key]) / denom)
        ax.bar(x + offset, values, width=width, color=color, label=label, edgecolor="black", linewidth=0.4)
    ax.axhline(1.0, color="0.35", lw=0.8, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in metric_specs])
    ax.set_ylabel("Relative value (MLP = 1)")
    ax.set_title("(d) Final task-quality metrics")
    ymax = max(
        float(quality[label][key]) / max(float(baseline[key]), 1e-8)
        for label, _, _ in METHODS
        for key, _ in metric_specs
    )
    ax.set_ylim(0, max(1.35, ymax * 1.18))
    ax.legend(frameon=False, loc="upper left", fontsize=7)
    polish_axis(ax)


def make_figure(path, metrics, examples, quality, with_title=True):
    fig = plt.figure(figsize=(12.0, 8.6), dpi=300)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.18], width_ratios=[1.28, 1.0], hspace=0.34, wspace=0.16)
    ax = fig.add_subplot(gs[0, 0])
    plot_reward(ax, metrics)
    ax = fig.add_subplot(gs[0, 1])
    plot_length(ax, metrics)
    plot_aux_examples(gs[1, 0], fig, examples)
    ax = fig.add_subplot(gs[1, 1])
    plot_final_quality(ax, quality)
    if with_title:
        fig.suptitle("GRU ablation and auxiliary future-motion prediction", fontsize=13, y=0.985)
        fig.subplots_adjust(left=0.055, right=0.985, top=0.92, bottom=0.07)
    else:
        fig.subplots_adjust(left=0.055, right=0.985, top=0.97, bottom=0.07)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    metrics = load_metrics()
    examples, _ = choose_aux_examples()
    quality = collect_final_quality_metrics()
    make_figure(OUT_FIG, metrics, examples, quality, with_title=True)
    make_figure(OUT_FIG_NOTITLE, metrics, examples, quality, with_title=False)
    make_figure(OUT_FIG_FILLED, metrics, examples, quality, with_title=True)
    make_figure(OUT_FIG_FILLED_NOTITLE, metrics, examples, quality, with_title=False)
    for path in [OUT_FIG, OUT_FIG_NOTITLE, OUT_FIG_FILLED, OUT_FIG_FILLED_NOTITLE]:
        print(path.as_posix())
        print(path.with_suffix(".pdf").as_posix())


if __name__ == "__main__":
    main()
