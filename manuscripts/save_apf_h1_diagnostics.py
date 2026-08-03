from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import imageio.v2 as imageio
import numpy as np
import pygame
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "manuscripts"))

from src.custom_env import RehabilitationEnv
from src.renderer import render_aesthetic
from run_apf_table2_baseline import H1_PATH, ReactiveAPFController, seed_everything, spawn_seed_plan

OUT_DIR = ROOT / "manuscripts" / "current_league_zpd35_55_noid_warm_entropy_10iter_final" / "apf_table2_baseline" / "h1_diagnostics"
MAX_STEPS = 100
BASE_SEED = 2026
TRIALS = 10
EPISODES_PER_TRIAL = 100
TARGET_TIZ = 0.22732
TARGET_LENGTH = 41.108
FPS = 10
CONTROLLER_CONFIG = {
    "radial_gain": 1.20,
    "tangent_gain": 0.30,
    "boundary_gain": 1.35,
    "boundary_band": 1.25,
    "boundary_guard": 0.20,
    "smoothing": 0.10,
}


def make_env(hand_model: PPO) -> RehabilitationEnv:
    env = RehabilitationEnv(
        training_mode="robot",
        hand_model=hand_model,
        history_length=16,
        history_mode="motion",
        render_mode=None,
    )
    env.random_noise = False
    env.max_steps = MAX_STEPS
    env.scripted_hand_sample_prob = 0.0
    return env


def rollout(hand_model: PPO, seed: int, collect_frames: bool = False):
    env = make_env(hand_model)
    controller = ReactiveAPFController(**CONTROLLER_CONFIG)
    frames = []
    distance_rows = []
    done_reason = ""
    try:
        seed_everything(seed)
        env.reset(seed=seed)
        controller.reset()
        window = None
        if collect_frames:
            pygame.init()
            width_px = int(env.grid_size * env.cell_size * 1.5)
            height_px = int(env.grid_size * env.cell_size)
            window = pygame.display.set_mode((width_px, height_px))
            surface = render_aesthetic(env.robot_position, env.hand_position, env.fixed_point, env.trajectory_points, env.grid_size, env.cell_size, window)
            add_overlay(surface, seed, 0, float(env.current_distance), "")
            frames.append(surface_to_frame(surface))

        terminated = False
        truncated = False
        step = 0
        while not (terminated or truncated):
            action = controller.predict(env)
            _, _, terminated, truncated, info = env.step(action)
            step += 1
            dist = float(info.get("dist", 0.0))
            done_reason = str(info.get("done_reason", ""))
            distance_rows.append({
                "step": step,
                "distance": dist,
                "robot_x": float(env.robot_position[0]),
                "robot_y": float(env.robot_position[1]),
                "hand_x": float(env.hand_position[0]),
                "hand_y": float(env.hand_position[1]),
                "action_x": float(action[0]),
                "action_y": float(action[1]),
                "done_reason": done_reason if (terminated or truncated) else "",
            })
            if collect_frames:
                surface = render_aesthetic(env.robot_position, env.hand_position, env.fixed_point, env.trajectory_points, env.grid_size, env.cell_size, window)
                add_overlay(surface, seed, step, dist, done_reason if (terminated or truncated) else "")
                frames.append(surface_to_frame(surface))

        if collect_frames and frames:
            for _ in range(max(5, FPS // 2)):
                frames.append(frames[-1])
    finally:
        env.close()
        if collect_frames:
            pygame.display.quit()
            pygame.quit()

    distances = np.asarray([row["distance"] for row in distance_rows], dtype=float)
    in_zpd = (distances >= 3.5) & (distances <= 5.5) if distances.size else np.asarray([], dtype=bool)
    return {
        "seed": int(seed),
        "episode_length": int(len(distance_rows)),
        "tiz": float(np.sum(in_zpd) / MAX_STEPS) if distances.size else 0.0,
        "zpd_coverage": float(np.mean(in_zpd)) if distances.size else 0.0,
        "too_close_rate": float(np.mean(distances < 3.5)) if distances.size else 0.0,
        "too_far_rate": float(np.mean(distances > 5.5)) if distances.size else 0.0,
        "avg_distance": float(np.mean(distances)) if distances.size else 0.0,
        "done_reason": done_reason,
        "distance_rows": distance_rows,
        "frames": frames,
    }


def add_overlay(surface, seed: int, step: int, distance: float, done_reason: str):
    font = pygame.font.SysFont("arial", 18)
    lines = [
        f"Reactive APF vs H1  seed={seed}",
        f"step={step:03d}  distance={distance:.2f}  ZPD=[3.5, 5.5]",
    ]
    if done_reason:
        lines.append(f"termination: {done_reason}")
    for idx, line in enumerate(lines):
        text = font.render(line, True, (30, 30, 30))
        bg = pygame.Surface((text.get_width() + 12, text.get_height() + 8), pygame.SRCALPHA)
        bg.fill((255, 255, 255, 210))
        surface.blit(bg, (10, 10 + idx * 28))
        surface.blit(text, (16, 14 + idx * 28))


def surface_to_frame(surface) -> np.ndarray:
    arr = pygame.surfarray.array3d(surface)
    return np.transpose(arr, (1, 0, 2)).copy()


def write_distance_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    hand_model = PPO.load(str(H1_PATH), custom_objects={"learning_rate": 0.0, "optimizer_class": None}, verbose=0)
    seed_plan = spawn_seed_plan(BASE_SEED, TRIALS, EPISODES_PER_TRIAL)
    seeds = [seed for _, episode_seeds in seed_plan for seed in episode_seeds]
    rows = []
    for seed in seeds:
        result = rollout(hand_model, seed, collect_frames=False)
        result.pop("distance_rows")
        result.pop("frames")
        rows.append(result)

    rows.sort(key=lambda row: row["tiz"], reverse=True)
    high = rows[0]
    representative = min(
        rows,
        key=lambda row: abs(row["tiz"] - TARGET_TIZ) / TARGET_TIZ + abs(row["episode_length"] - TARGET_LENGTH) / TARGET_LENGTH,
    )
    ranking_path = OUT_DIR / "apf_h1_episode_ranking_20260722.csv"
    with ranking_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summaries = []
    for label, selected in [("representative", representative), ("highest_tiz", high)]:
        result = rollout(hand_model, int(selected["seed"]), collect_frames=True)
        video_path = OUT_DIR / f"reactive_apf_h1_{label}_seed{result['seed']}.mp4"
        trace_path = OUT_DIR / f"reactive_apf_h1_{label}_seed{result['seed']}_distance.csv"
        imageio.mimsave(video_path, result["frames"], fps=FPS, quality=8, macro_block_size=2)
        write_distance_csv(trace_path, result["distance_rows"])
        summary = {k: v for k, v in result.items() if k not in {"frames", "distance_rows"}}
        summary["label"] = label
        summary["video_path"] = str(video_path)
        summary["distance_csv"] = str(trace_path)
        summaries.append(summary)
        print(summary)

    summary_path = OUT_DIR / "apf_h1_video_summary_20260722.json"
    summary_path.write_text(json.dumps({"ranking_csv": str(ranking_path), "summaries": summaries}, indent=2), encoding="utf-8")
    print(f"ranking: {ranking_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
