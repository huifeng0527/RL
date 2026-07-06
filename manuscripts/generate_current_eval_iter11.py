from pathlib import Path
import json
import re
from dataclasses import asdict, dataclass

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from stable_baselines3 import PPO

from src.custom_env import RehabilitationEnv
from src.observation_schema import OBS_SCALAR_DIM, HISTORY_CHANNELS, INTERACTION_HISTORY_CHANNELS, obs_dim, adapt_history_obs, model_obs_dim
from src.scripts.test_env import SluggishScriptHand, SpasmScriptHand

BASE = Path(r"C:/Users/admin/Desktop/research/RL")
RUN_DIR = BASE / "logs" / "league_paper_gru_multistep_aux_20iter"
OUT_DIR = BASE / "manuscripts" / "current_eval_iter11"
OUT_DIR.mkdir(exist_ok=True)

BASELINE_A = BASE / "logs" / "ablation_study_0607_1530" / "2_MLP_LSTM" / "best_model.zip"
BASELINE_B = BASE / "logs" / "dual_iterative_0427_1314" / "baseline_b" / "robot" / "best_model.zip"
PFSP_ROBOT = RUN_DIR / "iteration_11" / "robot" / "robot" / "best_model.zip"
UNSEEN_HAND = RUN_DIR / "iteration_10" / "hand" / "hand" / "final_model.zip"

CROSS_EPISODES = 20
BASELINE_EPISODES = 30
MAX_STEPS = 100
ZPD_MIN = 4.0
ZPD_MAX = 6.0


@dataclass
class Metrics:
    mean_tis: float
    mean_zpd_coverage: float
    mean_episode_length: float
    mean_distance: float
    catch_rate: float
    out_rate: float
    timeout_rate: float
    episodes: int


def infer_history_mode_for_model(model):
    dim = model_obs_dim(model) or 44
    interaction_dim = obs_dim(16, 0, INTERACTION_HISTORY_CHANNELS)
    return "interaction" if dim >= interaction_dim else "motion"


def load_ppo(path):
    return PPO.load(str(path), custom_objects={"learning_rate": 0.0, "optimizer_class": None}, verbose=0)


def choose_model(iteration, kind):
    root = RUN_DIR / f"iteration_{iteration}" / kind / kind
    final = root / "final_model.zip"
    best = root / "best_model.zip"
    if final.exists():
        return final
    if best.exists():
        return best
    return None


def robot_models(max_iter=11):
    models = []
    for i in range(1, max_iter + 1):
        path = choose_model(i, "robot")
        if path is not None:
            models.append((f"R{i}", path))
    return models


def hand_models(max_iter=10):
    models = []
    for i in range(1, max_iter + 1):
        path = choose_model(i, "hand")
        if path is not None:
            models.append((f"H{i}", path))
    return models


def apply_script_hand(env, controller):
    desired_move = controller.get_move(env.hand_position, env.robot_position)
    alpha = 0.7
    smoothed_move = alpha * desired_move + (1 - alpha) * env.last_hand_actual_move
    max_accel = 0.15
    delta_v = smoothed_move - env.last_hand_actual_move
    accel_magnitude = np.linalg.norm(delta_v)
    if accel_magnitude > max_accel:
        delta_v = (delta_v / accel_magnitude) * max_accel
    final_move = env.last_hand_actual_move + delta_v
    env.last_hand_actual_move = final_move.copy()
    env.hand_position += final_move
    env.hand_position = np.clip(
        env.hand_position,
        env.margin,
        [env.env_width - env.margin, env.env_height - env.margin],
    )
    env.hand_history_buffer.append(final_move)


def evaluate_robot(robot_path, episodes=20, hand_path=None, test_type="rl_hand"):
    robot = load_ppo(robot_path)
    history_mode = infer_history_mode_for_model(robot)
    hand_model = load_ppo(hand_path) if hand_path is not None else None
    env = RehabilitationEnv(
        training_mode="robot",
        hand_model=hand_model,
        history_length=16,
        history_mode=history_mode,
    )
    env.random_noise = False
    env.max_steps = MAX_STEPS
    env.scripted_hand_sample_prob = 0.0

    controller = None
    if test_type == "sluggish":
        controller = SluggishScriptHand()
        env._bypass_hand_physics = True
    elif test_type == "spasm":
        controller = SpasmScriptHand()
        env._bypass_hand_physics = True
    elif test_type == "scripted":
        hand_model = None
        env.hand_model = None
        env.scripted_hand_sample_prob = 1.0
    elif test_type == "rl_hand":
        if hand_model is None:
            raise ValueError("rl_hand test requires hand_path")
    else:
        raise ValueError(test_type)

    tis_values = []
    zpd_values = []
    lengths = []
    distances_all = []
    catches = 0
    outs = 0
    timeouts = 0

    for _ in range(episodes):
        obs, info = env.reset()
        if isinstance(controller, SpasmScriptHand):
            controller.reset()
        dist_hist = []
        done = False
        truncated = False
        last_info = info

        while not (done or truncated):
            if controller is not None:
                apply_script_hand(env, controller)
            action, _ = robot.predict(obs, deterministic=True)
            obs, _, done, truncated, last_info = env.step(action)
            dist_hist.append(float(last_info.get("dist", 0.0)))

        distances = np.asarray(dist_hist, dtype=float)
        if len(distances):
            in_zpd = (distances >= ZPD_MIN) & (distances <= ZPD_MAX)
            tis_values.append(float(np.sum(in_zpd) / MAX_STEPS))
            zpd_values.append(float(np.mean(in_zpd)))
            distances_all.extend(distances.tolist())
        else:
            tis_values.append(0.0)
            zpd_values.append(0.0)
        lengths.append(len(dist_hist))
        reason = last_info.get("done_reason")
        if reason == "Robot Caught":
            catches += 1
        elif reason == "Robot Out":
            outs += 1
        elif truncated:
            timeouts += 1

    env.close()
    return Metrics(
        mean_tis=float(np.mean(tis_values)),
        mean_zpd_coverage=float(np.mean(zpd_values)),
        mean_episode_length=float(np.mean(lengths)),
        mean_distance=float(np.mean(distances_all)) if distances_all else 0.0,
        catch_rate=float(catches / episodes),
        out_rate=float(outs / episodes),
        timeout_rate=float(timeouts / episodes),
        episodes=int(episodes),
    )


def run_cross_eval():
    robots = robot_models(11)
    hands = hand_models(10)
    tis = np.zeros((len(robots), len(hands)), dtype=float)
    zpd = np.zeros_like(tis)
    length = np.zeros_like(tis)
    records = []
    for i, (r_name, r_path) in enumerate(robots):
        for j, (h_name, h_path) in enumerate(hands):
            m = evaluate_robot(r_path, episodes=CROSS_EPISODES, hand_path=h_path, test_type="rl_hand")
            tis[i, j] = m.mean_tis
            zpd[i, j] = m.mean_zpd_coverage
            length[i, j] = m.mean_episode_length
            records.append({
                "robot": r_name,
                "robot_path": str(r_path),
                "hand": h_name,
                "hand_path": str(h_path),
                **asdict(m),
            })
            print(f"cross {r_name} vs {h_name}: TIS={m.mean_tis:.3f}, ZPD={m.mean_zpd_coverage:.3f}, Len={m.mean_episode_length:.1f}")

    payload = {
        "run_dir": str(RUN_DIR),
        "episodes_per_pair": CROSS_EPISODES,
        "robots": [{"name": n, "path": str(p)} for n, p in robots],
        "hands": [{"name": n, "path": str(p)} for n, p in hands],
        "tis_matrix": tis.tolist(),
        "zpd_matrix": zpd.tolist(),
        "episode_length_matrix": length.tolist(),
        "records": records,
    }
    (OUT_DIR / "cross_iter_validation_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9.5, 7.2), dpi=180)
    sns.heatmap(
        tis,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=max(0.7, float(np.nanmax(tis)) if tis.size else 0.7),
        xticklabels=[n for n, _ in hands],
        yticklabels=[n for n, _ in robots],
        cbar_kws={"label": "TIS ↑"},
    )
    ax.set_xlabel("Hand generation")
    ax.set_ylabel("Robot generation")
    ax.set_title(f"Cross-iteration validation heatmap\nTIS, {CROSS_EPISODES} episodes per robot-hand pair")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_cross_iter_validation_tis.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 7.2), dpi=180)
    sns.heatmap(
        zpd,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        vmin=0.0,
        vmax=max(0.8, float(np.nanmax(zpd)) if zpd.size else 0.8),
        xticklabels=[n for n, _ in hands],
        yticklabels=[n for n, _ in robots],
        cbar_kws={"label": "ZPD coverage ↑"},
    )
    ax.set_xlabel("Hand generation")
    ax.set_ylabel("Robot generation")
    ax.set_title(f"Cross-iteration validation heatmap\nZPD coverage, {CROSS_EPISODES} episodes per pair")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_cross_iter_validation_zpd.png")
    plt.close(fig)


def run_baseline_compare():
    robots = {
        "Baseline A": BASELINE_A,
        "Baseline B": BASELINE_B,
        "PFSP Ours": PFSP_ROBOT,
    }
    tests = {
        "Sluggish": {"test_type": "sluggish", "hand_path": None},
        "Spasm": {"test_type": "spasm", "hand_path": None},
        "Unseen RL hand": {"test_type": "rl_hand", "hand_path": UNSEEN_HAND},
    }
    results = []
    skipped = []
    for robot_name, robot_path in robots.items():
        if not robot_path.exists():
            print(f"missing robot {robot_name}: {robot_path}")
            skipped.append({"robot": robot_name, "path": str(robot_path), "reason": "missing file"})
            continue
        try:
            _ = load_ppo(robot_path)
        except Exception as exc:
            reason = str(exc).split("\n")[0]
            print(f"skip {robot_name}: {reason}")
            skipped.append({"robot": robot_name, "path": str(robot_path), "reason": reason})
            continue
        for test_name, cfg in tests.items():
            if cfg["hand_path"] is not None and not cfg["hand_path"].exists():
                print(f"missing hand for {test_name}: {cfg['hand_path']}")
                continue
            m = evaluate_robot(robot_path, episodes=BASELINE_EPISODES, hand_path=cfg["hand_path"], test_type=cfg["test_type"])
            row = {"robot": robot_name, "test": test_name, "robot_path": str(robot_path), **asdict(m)}
            results.append(row)
            print(f"baseline {robot_name} / {test_name}: TIS={m.mean_tis:.3f}, ZPD={m.mean_zpd_coverage:.3f}, catch={m.catch_rate:.2f}")

    (OUT_DIR / "baseline_comparison_results.json").write_text(json.dumps({
        "episodes_per_test": BASELINE_EPISODES,
        "results": results,
        "skipped": skipped,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    robot_order = ["Baseline A", "Baseline B", "PFSP Ours"]
    test_order = list(tests.keys())
    metric_names = ["mean_tis", "mean_zpd_coverage", "catch_rate"]
    titles = ["TIS ↑", "ZPD coverage ↑", "Catch rate ↓"]
    colors = ["#4C78A8", "#72B7B2", "#E45756"]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=180)
    x = np.arange(len(test_order))
    width = 0.24
    for ax, metric, title, color in zip(axes, metric_names, titles, colors):
        for k, robot_name in enumerate(robot_order):
            vals = []
            for test_name in test_order:
                match = next((r for r in results if r["robot"] == robot_name and r["test"] == test_name), None)
                vals.append(match[metric] if match else np.nan)
            ax.bar(x + (k - 1) * width, vals, width=width, label=robot_name)
        ax.set_xticks(x)
        ax.set_xticklabels(test_order, rotation=20, ha="right")
        ax.set_ylim(0, 1.0 if metric != "mean_tis" else max(0.7, max([r[metric] for r in results]) + 0.1))
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        if ax is axes[0]:
            ax.legend(fontsize=8)
    fig.suptitle(f"Baseline A/B vs PFSP stress comparison\n{BASELINE_EPISODES} episodes per test; compatible obs auto-detection")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_DIR / "fig_baseline_ab_pfsp_comparison.png")
    plt.close(fig)


def main():
    run_cross_eval()
    run_baseline_compare()
    print(OUT_DIR)


if __name__ == "__main__":
    main()
