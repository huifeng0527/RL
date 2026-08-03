from __future__ import annotations

import argparse
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


OUT_DIR = ROOT / "manuscripts" / "current_league_zpd35_55_noid_warm_entropy_10iter_final" / "apf_table2_baseline" / "videos"


def make_env(test_type: str, hand_model: PPO | None, max_steps: int) -> RehabilitationEnv:
    env = RehabilitationEnv(
        training_mode="robot",
        hand_model=hand_model,
        history_length=16,
        history_mode="motion",
        render_mode=None,
    )
    env.random_noise = False
    env.max_steps = int(max_steps)
    env.scripted_hand_sample_prob = 1.0 if test_type == "scripted" else 0.0
    return env


def rollout_metrics(test_type: str, hand_model: PPO | None, seed: int, max_steps: int, controller_config: dict):
    env = make_env(test_type, hand_model, max_steps)
    controller = ReactiveAPFController(**controller_config)
    try:
        seed_everything(seed)
        env.reset(seed=seed)
        controller.reset()
        terminated = False
        truncated = False
        distances = []
        done_reason = ""
        while not (terminated or truncated):
            action = controller.predict(env)
            _, _, terminated, truncated, info = env.step(action)
            distances.append(float(info.get("dist", 0.0)))
            done_reason = str(info.get("done_reason", ""))
        dist_arr = np.asarray(distances, dtype=float)
        in_zpd = (dist_arr >= float(env.zpd_min)) & (dist_arr <= float(env.zpd_max)) if dist_arr.size else np.asarray([], dtype=bool)
        return {
            "seed": int(seed),
            "length": int(len(distances)),
            "tiz": float(np.sum(in_zpd) / max_steps) if dist_arr.size else 0.0,
            "done_reason": done_reason,
        }
    finally:
        env.close()


def choose_representative_seed(
    test_type: str,
    hand_model: PPO | None,
    base_seed: int,
    max_steps: int,
    target_length: float,
    target_tiz: float,
    controller_config: dict,
):
    plan = spawn_seed_plan(base_seed, trials=3, episodes_per_trial=100)
    episode_seeds = [seed for _, seeds in plan for seed in seeds]
    candidates = [rollout_metrics(test_type, hand_model, seed, max_steps, controller_config) for seed in episode_seeds]
    candidates.sort(
        key=lambda row: (
            abs(row["length"] - target_length) / max(target_length, 1.0)
            + 2.0 * abs(row["tiz"] - target_tiz) / max(target_tiz, 1e-6)
        )
    )
    return candidates[0], candidates[:5]


def add_overlay(surface, label: str, step: int, dist: float, z_min: float, z_max: float, done_reason: str = ""):
    font = pygame.font.SysFont("arial", 18)
    lines = [
        label,
        f"step={step:03d}  distance={dist:.2f}  ZPD=[{z_min:.1f}, {z_max:.1f}]",
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


def save_rollout_video(
    test_name: str,
    test_type: str,
    hand_model: PPO | None,
    seed: int,
    max_steps: int,
    fps: int,
    controller_config: dict,
    output_path: Path,
):
    pygame.init()
    env = make_env(test_type, hand_model, max_steps)
    controller = ReactiveAPFController(**controller_config)
    width_px = int(env.grid_size * env.cell_size * 1.5)
    height_px = int(env.grid_size * env.cell_size)
    window = pygame.display.set_mode((width_px, height_px))
    frames = []
    distances = []
    done_reason = ""
    try:
        seed_everything(seed)
        env.reset(seed=seed)
        controller.reset()
        surface = render_aesthetic(env.robot_position, env.hand_position, env.fixed_point, env.trajectory_points, env.grid_size, env.cell_size, window)
        add_overlay(surface, f"Reactive APF vs {test_name}", 0, float(env.current_distance), float(env.zpd_min), float(env.zpd_max))
        frames.append(surface_to_frame(surface))

        terminated = False
        truncated = False
        step = 0
        while not (terminated or truncated):
            action = controller.predict(env)
            _, _, terminated, truncated, info = env.step(action)
            step += 1
            dist = float(info.get("dist", 0.0))
            distances.append(dist)
            done_reason = str(info.get("done_reason", ""))
            surface = render_aesthetic(env.robot_position, env.hand_position, env.fixed_point, env.trajectory_points, env.grid_size, env.cell_size, window)
            add_overlay(
                surface,
                f"Reactive APF vs {test_name}",
                step,
                dist,
                float(env.zpd_min),
                float(env.zpd_max),
                done_reason if (terminated or truncated) else "",
            )
            frames.append(surface_to_frame(surface))

        for _ in range(max(5, fps // 2)):
            frames.append(frames[-1])
    finally:
        env.close()
        pygame.display.quit()
        pygame.quit()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps, quality=8, macro_block_size=2)

    dist_arr = np.asarray(distances, dtype=float)
    in_zpd = (dist_arr >= 3.5) & (dist_arr <= 5.5) if dist_arr.size else np.asarray([], dtype=bool)
    return {
        "video": str(output_path),
        "seed": int(seed),
        "frames": len(frames),
        "episode_length": int(len(distances)),
        "tiz": float(np.sum(in_zpd) / max_steps) if dist_arr.size else 0.0,
        "done_reason": done_reason,
    }


def main():
    parser = argparse.ArgumentParser(description="Save APF Table II rollout videos.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--out_dir", type=Path, default=OUT_DIR)
    parser.add_argument("--h1_path", type=Path, default=H1_PATH)
    parser.add_argument("--radial_gain", type=float, default=1.20)
    parser.add_argument("--tangent_gain", type=float, default=0.30)
    parser.add_argument("--boundary_gain", type=float, default=1.35)
    parser.add_argument("--boundary_band", type=float, default=1.25)
    parser.add_argument("--boundary_guard", type=float, default=0.20)
    parser.add_argument("--smoothing", type=float, default=0.10)
    args = parser.parse_args()

    if not args.h1_path.exists():
        raise FileNotFoundError(args.h1_path)

    controller_config = {
        "radial_gain": args.radial_gain,
        "tangent_gain": args.tangent_gain,
        "boundary_gain": args.boundary_gain,
        "boundary_band": args.boundary_band,
        "boundary_guard": args.boundary_guard,
        "smoothing": args.smoothing,
    }
    hand_model = PPO.load(str(args.h1_path), custom_objects={"learning_rate": 0.0, "optimizer_class": None}, verbose=0)

    conditions = [
        ("scripted_hand", "scripted", None, 18.1, 0.089),
        ("agent_H1", "learned", hand_model, 29.7, 0.173),
    ]
    summaries = []
    for test_name, test_type, model, target_length, target_tiz in conditions:
        selected, top = choose_representative_seed(
            test_type,
            model,
            args.seed,
            args.max_steps,
            target_length,
            target_tiz,
            controller_config,
        )
        print(f"{test_name}: selected seed {selected['seed']} length={selected['length']} TIZ={selected['tiz']:.3f}; candidates={top}")
        summary = save_rollout_video(
            test_name=test_name,
            test_type=test_type,
            hand_model=model,
            seed=selected["seed"],
            max_steps=args.max_steps,
            fps=args.fps,
            controller_config=controller_config,
            output_path=args.out_dir / f"reactive_apf_{test_name}_seed{selected['seed']}.mp4",
        )
        summaries.append(summary)
        print(summary)

    print("Saved videos:")
    for summary in summaries:
        print(summary["video"])


if __name__ == "__main__":
    main()
