from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
import torch as th
from stable_baselines3 import PPO

from src.custom_env import RehabilitationEnv
from src.observation_schema import OBS_SCALAR_DIM, HISTORY_CHANNELS, INTERACTION_HISTORY_CHANNELS, history_slice

RUN_DIR = Path(r"C:/Users/admin/Desktop/research/RL/logs/league_paper_gru_multistep_aux_20iter")
OUT_DIR = Path(r"C:/Users/admin/Desktop/research/RL/manuscripts/current_league_preview_8iter")
OUT_DIR.mkdir(exist_ok=True)

ROBOT_PATH = RUN_DIR / "iteration_9" / "robot" / "robot" / "final_model.zip"
HAND_PATH = RUN_DIR / "iteration_8" / "hand" / "hand" / "final_model.zip"
OUT_FIG = OUT_DIR / "fig_aux_multistep_prediction_current.png"
OUT_JSON = OUT_DIR / "aux_multistep_prediction_examples_current.json"

HISTORY_LENGTH = 16
FUTURE_HORIZON = 8
N_EPISODES = 80
MAX_EXAMPLES = 3


def get_future_prediction(model, obs):
    extractor = model.policy.features_extractor
    device = model.device
    obs_tensor = th.as_tensor(obs[None, :], dtype=th.float32, device=device)
    extractor.eval()
    with th.no_grad():
        pred_traj, pred_risk_logit = extractor.forward_aux_future(obs_tensor)
    return (
        pred_traj.detach().cpu().numpy()[0],
        float(th.sigmoid(pred_risk_logit).detach().cpu().numpy()[0]),
    )


def collect_examples(robot_model, hand_model):
    env = RehabilitationEnv(
        training_mode="robot",
        hand_model=hand_model,
        history_length=HISTORY_LENGTH,
        history_mode="interaction",
    )
    env.scripted_hand_sample_prob = 0.0
    env.random_noise = False

    h_slice = history_slice(HISTORY_LENGTH, INTERACTION_HISTORY_CHANNELS)
    latest_step_start = h_slice.stop - INTERACTION_HISTORY_CHANNELS
    hand_delta_start = latest_step_start + INTERACTION_HISTORY_CHANNELS - HISTORY_CHANNELS

    examples = []
    horizon_errors = []

    for _ in range(N_EPISODES):
        obs, _ = env.reset()
        obs_seq = []
        hand_pos_seq = []
        done = False
        truncated = False

        while not (done or truncated):
            obs_seq.append(np.asarray(obs, dtype=np.float32).copy())
            hand_pos_seq.append(np.asarray(env.hand_position, dtype=np.float32).copy())
            action, _ = robot_model.predict(obs, deterministic=True)
            obs, _, done, truncated, _ = env.step(action)

        if len(obs_seq) <= HISTORY_LENGTH + FUTURE_HORIZON + 2:
            continue

        obs_arr = np.asarray(obs_seq, dtype=np.float32)
        hand_pos_arr = np.asarray(hand_pos_seq, dtype=np.float32)
        for t in range(HISTORY_LENGTH, len(obs_arr) - FUTURE_HORIZON):
            pred_moves, pred_risk = get_future_prediction(robot_model, obs_arr[t])
            future = obs_arr[t + 1:t + 1 + FUTURE_HORIZON]
            true_moves = future[:, hand_delta_start:h_slice.stop]
            pred_traj = np.cumsum(pred_moves, axis=0)
            true_traj = np.cumsum(true_moves, axis=0)
            disp_errors = np.mean((pred_moves - true_moves) ** 2, axis=1)
            traj_errors = np.mean((pred_traj - true_traj) ** 2, axis=1)
            horizon_errors.append(traj_errors)

            if len(examples) < 80:
                examples.append({
                    "t": int(t),
                    "pred_moves": pred_moves,
                    "true_moves": true_moves,
                    "past_positions": hand_pos_arr[max(0, t - HISTORY_LENGTH + 1):t + 1],
                    "current_position": hand_pos_arr[t],
                    "future_positions": hand_pos_arr[t + 1:t + 1 + FUTURE_HORIZON],
                    "pred_risk": pred_risk,
                    "trajectory_mse": float(np.mean(traj_errors)),
                    "displacement_mse": float(np.mean(disp_errors)),
                })

    env.close()

    if not examples:
        raise RuntimeError("No valid rollout window collected for auxiliary prediction visualization.")

    examples = sorted(examples, key=lambda x: x["trajectory_mse"])
    if len(examples) >= MAX_EXAMPLES:
        idxs = np.linspace(0, len(examples) - 1, MAX_EXAMPLES + 2, dtype=int)[1:-1]
        chosen = [examples[i] for i in idxs]
    else:
        chosen = examples

    return chosen, np.asarray(horizon_errors, dtype=np.float32)


def relative_xy(example):
    current = example["current_position"]
    past = example["past_positions"] - current
    true_future = example["future_positions"] - current
    pred_future = np.cumsum(example["pred_moves"], axis=0)
    target_future_from_moves = np.cumsum(example["true_moves"], axis=0)
    return past, true_future, pred_future, target_future_from_moves


def plot_examples(examples, horizon_errors):
    n = len(examples)
    fig = plt.figure(figsize=(12.5, 3.4 * n), dpi=180)
    gs = fig.add_gridspec(n, 2, width_ratios=[1.25, 1.0], hspace=0.42, wspace=0.28)

    for row, ex in enumerate(examples):
        ax = fig.add_subplot(gs[row, 0])
        past, true_future, pred_future, target_future_from_moves = relative_xy(ex)

        ax.plot(past[:, 0], past[:, 1], "o-", color="#9E9E9E", linewidth=1.5, markersize=3, label="past hand path")
        ax.scatter([0], [0], marker="*", s=110, color="black", label="current")
        ax.plot(target_future_from_moves[:, 0], target_future_from_moves[:, 1], "o-", color="#4C78A8", linewidth=2.0, markersize=4, label="ground-truth future trajectory")
        ax.plot(pred_future[:, 0], pred_future[:, 1], "s--", color="#F58518", linewidth=2.0, markersize=4, label="predicted future trajectory")

        for k, p in enumerate(pred_future, start=1):
            ax.text(p[0], p[1], str(k), fontsize=7, color="#8C4B00")

        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.25)
        ax.set_xlabel("relative x")
        ax.set_ylabel("relative y")
        ax.set_title(f"Example {row + 1}: trajectory prediction, Traj. MSE={ex['trajectory_mse']:.4f}, Disp. MSE={ex['displacement_mse']:.4f}")
        if row == 0:
            ax.legend(fontsize=7, loc="best", frameon=True)

        ax2 = fig.add_subplot(gs[row, 1])
        true_traj = target_future_from_moves
        pred_traj = pred_future
        horizons = np.arange(1, FUTURE_HORIZON + 1)
        ax2.plot(horizons, true_traj[:, 0], "o-", color="#4C78A8", label="true x")
        ax2.plot(horizons, pred_traj[:, 0], "o--", color="#F58518", label="pred x")
        ax2.plot(horizons, true_traj[:, 1], "s-", color="#72B7B2", label="true y")
        ax2.plot(horizons, pred_traj[:, 1], "s--", color="#E45756", label="pred y")
        ax2.axhline(0, color="black", linewidth=0.8, alpha=0.35)
        ax2.set_xlabel("future step")
        ax2.set_ylabel("cumulative relative position")
        ax2.set_title("Trajectory coordinates by horizon")
        ax2.grid(alpha=0.25)
        if row == 0:
            ax2.legend(fontsize=7, ncol=2, frameon=True)

    if horizon_errors.size:
        inset = fig.add_axes([0.70, 0.02, 0.25, 0.10])
        horizons = np.arange(1, FUTURE_HORIZON + 1)
        mean = np.mean(horizon_errors, axis=0)
        stderr = np.std(horizon_errors, axis=0) / np.sqrt(max(len(horizon_errors), 1))
        inset.plot(horizons, mean, "o-", color="#B279A2")
        inset.fill_between(horizons, mean - stderr, mean + stderr, color="#B279A2", alpha=0.2)
        inset.set_title("Trajectory MSE by horizon", fontsize=8)
        inset.set_xlabel("step", fontsize=7)
        inset.set_ylabel("Traj. MSE", fontsize=7)
        inset.tick_params(labelsize=7)
        inset.grid(alpha=0.2)

    fig.suptitle("Auxiliary multi-step forward prediction visualization\nGRU + multi-risk auxiliary robot checkpoint, current rollout samples", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0, 0.06, 1, 0.965))
    fig.savefig(OUT_FIG)
    plt.close(fig)


def main():
    robot_model = PPO.load(str(ROBOT_PATH), verbose=0)
    hand_model = PPO.load(str(HAND_PATH), custom_objects={"learning_rate": 0.0, "optimizer_class": None}, verbose=0)
    examples, horizon_errors = collect_examples(robot_model, hand_model)
    plot_examples(examples, horizon_errors)

    serializable = []
    for ex in examples:
        serializable.append({
            "t": ex["t"],
            "trajectory_mse": ex["trajectory_mse"],
            "displacement_mse": ex["displacement_mse"],
            "pred_risk": ex["pred_risk"],
            "pred_moves": ex["pred_moves"].tolist(),
            "true_moves": ex["true_moves"].tolist(),
        })
    OUT_JSON.write_text(json.dumps({
        "robot_path": str(ROBOT_PATH),
        "hand_path": str(HAND_PATH),
        "future_horizon": FUTURE_HORIZON,
        "examples": serializable,
        "trajectory_mse_by_horizon_mean": np.mean(horizon_errors, axis=0).tolist() if horizon_errors.size else [],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(OUT_FIG)
    print(OUT_JSON)


if __name__ == "__main__":
    main()
