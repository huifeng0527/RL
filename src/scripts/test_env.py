"""Final comparison evaluator for scripted-only, single-hand, and league robots.

Compares:
- Baseline A: robot trained against scripted hand only
- Baseline B: robot trained against one frozen learned hand
- PFSP/League: robot trained with iterative league sampling

Default paths target the current league run. Use command-line arguments to replace
any model path or learned-hand test set.
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
import numpy as np
import pygame
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.custom_env import RehabilitationEnv
from src.observation_schema import INTERACTION_HISTORY_CHANNELS, model_obs_dim, obs_dim
from src.renderer import render_aesthetic


DEFAULT_BASE_DIR = r"C:\Users\admin\Desktop\research\RL\logs\league_paper_gru_multistep_aux_pfsp_window_20iter"

ZPD_MIN = 3.5
ZPD_MAX = 5.5


@dataclass
class EvalMetrics:
    robot_name: str
    robot_path: str
    test_name: str
    test_type: str
    hand_path: str | None
    episodes: int
    max_steps: int
    reward_mean: float
    reward_std: float
    tis_mean: float
    tis_std: float
    zpd_coverage_mean: float
    zpd_coverage_std: float
    episode_length_mean: float
    episode_length_std: float
    catch_rate: float
    too_close_rate: float
    too_far_rate: float
    avg_distance_mean: float
    avg_distance_std: float


def learned_hand_path(base_dir: str, generation: int):
    return os.path.join(base_dir, f"iteration_{generation}", "hand", "hand", "best_model.zip")


def infer_robot_history_mode(robot_model: PPO, history_length: int):
    expected_dim = model_obs_dim(robot_model)
    interaction_dim = obs_dim(history_length, 0, INTERACTION_HISTORY_CHANNELS)
    return "interaction" if expected_dim == interaction_dim else "motion"


def make_env(robot_model: PPO, hand_model: PPO | None, history_length: int, scripted_prob: float):
    history_mode = infer_robot_history_mode(robot_model, history_length)
    env = RehabilitationEnv(
        training_mode="robot",
        hand_model=hand_model,
        history_length=history_length,
        history_mode=history_mode,
    )
    env.scripted_hand_sample_prob = scripted_prob
    env.random_noise = False
    return env


def evaluate_robot_on_test(
    robot_name: str,
    robot_path: str,
    test_name: str,
    test_type: str,
    hand_path: str | None,
    episodes: int,
    max_steps: int,
    history_length: int,
    seed: int,
):
    robot_model = PPO.load(robot_path, verbose=0)
    hand_model = None
    if test_type == "learned":
        hand_model = PPO.load(
            hand_path,
            custom_objects={"learning_rate": 0.0, "optimizer_class": None},
            verbose=0,
        )

    env = make_env(
        robot_model=robot_model,
        hand_model=hand_model,
        history_length=history_length,
        scripted_prob=0.0 if test_type == "learned" else 1.0,
    )
    env.max_steps = max_steps

    rewards = []
    tis_scores = []
    zpd_coverages = []
    episode_lengths = []
    catch_flags = []
    too_close_rates = []
    too_far_rates = []
    avg_distances = []

    for ep in range(episodes):
        np.random.seed(seed + ep)
        obs, _ = env.reset(seed=seed + ep)

        terminated = False
        truncated = False
        total_reward = 0.0
        distances = []
        done_reason = ""

        while not (terminated or truncated):
            action, _ = robot_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            distances.append(float(info.get("dist", 0.0)))
            done_reason = str(info.get("done_reason", ""))

        dist_arr = np.asarray(distances, dtype=np.float32)
        episode_len = len(distances)
        zpd_min = float(env.zpd_min)
        zpd_max = float(env.zpd_max)
        in_zpd = (dist_arr >= zpd_min) & (dist_arr <= zpd_max) if episode_len else np.asarray([], dtype=bool)
        rewards.append(total_reward)
        tis_scores.append(float(np.sum(in_zpd) / max_steps) if episode_len else 0.0)
        zpd_coverages.append(float(np.mean(in_zpd)) if episode_len else 0.0)
        episode_lengths.append(episode_len)
        catch_flags.append(done_reason == "Robot Caught")
        too_close_rates.append(float(np.mean(dist_arr < zpd_min)) if episode_len else 0.0)
        too_far_rates.append(float(np.mean(dist_arr > zpd_max)) if episode_len else 0.0)
        avg_distances.append(float(np.mean(dist_arr)) if episode_len else 0.0)

    env.close()

    return EvalMetrics(
        robot_name=robot_name,
        robot_path=robot_path,
        test_name=test_name,
        test_type=test_type,
        hand_path=hand_path,
        episodes=episodes,
        max_steps=max_steps,
        reward_mean=float(np.mean(rewards)),
        reward_std=float(np.std(rewards)),
        tis_mean=float(np.mean(tis_scores)),
        tis_std=float(np.std(tis_scores)),
        zpd_coverage_mean=float(np.mean(zpd_coverages)),
        zpd_coverage_std=float(np.std(zpd_coverages)),
        episode_length_mean=float(np.mean(episode_lengths)),
        episode_length_std=float(np.std(episode_lengths)),
        catch_rate=float(np.mean(catch_flags)),
        too_close_rate=float(np.mean(too_close_rates)),
        too_far_rate=float(np.mean(too_far_rates)),
        avg_distance_mean=float(np.mean(avg_distances)),
        avg_distance_std=float(np.std(avg_distances)),
    )


def run_mouse_hand_test(
    robot_name: str,
    robot_path: str,
    episodes: int,
    max_steps: int,
    history_length: int,
    seed: int,
    fps: int,
):
    robot_model = PPO.load(robot_path, verbose=0)
    env = make_env(
        robot_model=robot_model,
        hand_model=None,
        history_length=history_length,
        scripted_prob=1.0,
    )
    env.max_steps = max_steps
    env.random_noise = False
    env.distance_threshold_collision = 1.5
    env.stride_hand = 0.3
    env.stride_robot = 0.6

    pygame.init()
    width_px = int(env.grid_size * env.cell_size * 1.5)
    height_px = int(env.grid_size * env.cell_size)
    screen = pygame.display.set_mode((width_px, height_px))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 18)

    current_mouse_move = {"value": np.zeros(2, dtype=np.float32)}
    original_resolve_hand_move = env._resolve_hand_move
    env._resolve_hand_move = lambda action: current_mouse_move["value"].copy()

    rewards = []
    tis_scores = []
    zpd_coverages = []
    episode_lengths = []
    catch_flags = []
    too_close_rates = []
    too_far_rates = []
    avg_distances = []

    try:
        for ep in range(episodes):
            np.random.seed(seed + ep)
            obs, _ = env.reset(seed=seed + ep)
            env.stride_hand = 0.5
            env.stride_robot = 0.6
            pygame.display.set_caption(f"Mouse hand test: {robot_name} | Episode {ep + 1}/{episodes}")

            terminated = False
            truncated = False
            quit_requested = False
            total_reward = 0.0
            distances = []
            done_reason = ""

            print(f"\nMouse hand episode {ep + 1}/{episodes}: control the HAND with the mouse. Press Q to quit, R to restart episode.")

            while not (terminated or truncated or quit_requested):
                restart_requested = False
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        quit_requested = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            quit_requested = True
                        elif event.key == pygame.K_r:
                            restart_requested = True
                if restart_requested:
                    break

                mouse_x, mouse_y = pygame.mouse.get_pos()
                target = np.array([
                    np.clip(mouse_x / env.cell_size, env.margin, env.env_width - env.margin),
                    np.clip(mouse_y / env.cell_size, env.margin, env.env_height - env.margin),
                ], dtype=np.float32)
                vec = target - env.hand_position
                dist = float(np.linalg.norm(vec))
                if dist > 1e-8:
                    current_mouse_move["value"] = (vec / dist * min(dist, env.stride_hand)).astype(np.float32)
                else:
                    current_mouse_move["value"] = np.zeros(2, dtype=np.float32)

                action, _ = robot_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                distances.append(float(info.get("dist", 0.0)))
                done_reason = str(info.get("done_reason", ""))

                render_aesthetic(
                    env.robot_position,
                    env.hand_position,
                    env.fixed_point,
                    env.trajectory_points,
                    grid_size=env.grid_size,
                    cell_size=env.cell_size,
                    window=screen,
                )
                text = font.render(
                    f"{robot_name} | ep {ep + 1}/{episodes} | dist {info.get('dist', 0):.2f} | step {len(distances)}/{max_steps}",
                    True,
                    (0, 0, 0),
                )
                screen.blit(text, (10, 10))
                pygame.display.flip()
                clock.tick(fps)

            if quit_requested:
                break

            dist_arr = np.asarray(distances, dtype=np.float32)
            episode_len = len(distances)
            zpd_min = float(env.zpd_min)
            zpd_max = float(env.zpd_max)
            in_zpd = (dist_arr >= zpd_min) & (dist_arr <= zpd_max) if episode_len else np.asarray([], dtype=bool)
            rewards.append(total_reward)
            tis_scores.append(float(np.sum(in_zpd) / max_steps) if episode_len else 0.0)
            zpd_coverages.append(float(np.mean(in_zpd)) if episode_len else 0.0)
            episode_lengths.append(episode_len)
            catch_flags.append(done_reason == "Robot Caught")
            too_close_rates.append(float(np.mean(dist_arr < zpd_min)) if episode_len else 0.0)
            too_far_rates.append(float(np.mean(dist_arr > zpd_max)) if episode_len else 0.0)
            avg_distances.append(float(np.mean(dist_arr)) if episode_len else 0.0)
    finally:
        env._resolve_hand_move = original_resolve_hand_move
        env.close()
        pygame.quit()

    completed = len(episode_lengths)
    if completed == 0:
        completed = 1
        rewards = [0.0]
        tis_scores = [0.0]
        zpd_coverages = [0.0]
        episode_lengths = [0]
        catch_flags = [False]
        too_close_rates = [0.0]
        too_far_rates = [0.0]
        avg_distances = [0.0]

    return EvalMetrics(
        robot_name=robot_name,
        robot_path=robot_path,
        test_name="mouse_hand",
        test_type="mouse",
        hand_path=None,
        episodes=completed,
        max_steps=max_steps,
        reward_mean=float(np.mean(rewards)),
        reward_std=float(np.std(rewards)),
        tis_mean=float(np.mean(tis_scores)),
        tis_std=float(np.std(tis_scores)),
        zpd_coverage_mean=float(np.mean(zpd_coverages)),
        zpd_coverage_std=float(np.std(zpd_coverages)),
        episode_length_mean=float(np.mean(episode_lengths)),
        episode_length_std=float(np.std(episode_lengths)),
        catch_rate=float(np.mean(catch_flags)),
        too_close_rate=float(np.mean(too_close_rates)),
        too_far_rate=float(np.mean(too_far_rates)),
        avg_distance_mean=float(np.mean(avg_distances)),
        avg_distance_std=float(np.std(avg_distances)),
    )


def parse_generations(raw: str):
    if not raw.strip():
        return []
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def build_tests(args):
    tests: list[tuple[str, str, str | None]] = []
    if args.test_set in {"scripted", "all"}:
        tests.append(("scripted_hand", "scripted", None))
    if args.test_set in {"agent", "learned", "all"}:
        generations = parse_generations(args.learned_hand_generations)
        for gen in generations:
            path = learned_hand_path(args.base_dir, gen)
            name = "agent_hand" if len(generations) == 1 else f"agent_H{gen}"
            tests.append((name, "learned", path))
    if args.test_set == "mouse":
        tests.append(("mouse_hand", "mouse", None))
    return tests


def default_robot_paths(base_dir: str):
    return {
        "scripted_only": os.path.join(base_dir, "baselines", "scripted_only_5m", "robot", "best_model.zip"),
        "single_h10": os.path.join(base_dir, "baselines", "single_hand_h1_5m", "robot", "best_model.zip"),
        "league": os.path.join(base_dir, "iteration_10", "robot", "robot", "best_model.zip"),
    }


def build_robots(args):
    defaults = default_robot_paths(args.base_dir)
    candidates = {
        "scripted_only": args.scripted_only_path or defaults["scripted_only"],
        "single_h10": args.single_hand_path or defaults["single_h10"],
        "league": args.league_path or defaults["league"],
    }
    selected = candidates if args.robot == "all" else {args.robot: candidates[args.robot]}
    return selected


def save_results(results: list[EvalMetrics], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"comparison_results_{timestamp}.json")
    csv_path = os.path.join(output_dir, f"comparison_results_{timestamp}.csv")

    rows = [asdict(r) for r in results]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")


def print_summary(results: list[EvalMetrics]):
    tests = []
    robots = []
    for item in results:
        if item.test_name not in tests:
            tests.append(item.test_name)
        if item.robot_name not in robots:
            robots.append(item.robot_name)

    by_key = {(r.robot_name, r.test_name): r for r in results}
    print("\n" + "=" * 100)
    print("FINAL POLICY COMPARISON: TIS / ZPD / Length / Catch")
    print("=" * 100)
    print(f"{'Robot':<16}" + "".join(f"{test:>24}" for test in tests))
    print("-" * 100)
    for robot in robots:
        row = f"{robot:<16}"
        for test in tests:
            r = by_key.get((robot, test))
            if r is None:
                row += f"{'N/A':>24}"
            else:
                cell = f"{r.tis_mean:.2f}/{r.zpd_coverage_mean:.2f}/{r.episode_length_mean:.0f}/{r.catch_rate:.0%}"
                row += f"{cell:>24}"
        print(row)
    print("=" * 100)
    print("Cell format: TIS / ZPD coverage / episode length / catch rate")


def main():
    parser = argparse.ArgumentParser(description="Evaluate robot policies on scripted hand, agent hand, and mouse-hand tests.")
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--robot", choices=["scripted_only", "single_h10", "league", "all"], default="all")
    parser.add_argument("--test_set", choices=["scripted", "agent", "learned", "mouse", "all"], default="all")
    parser.add_argument("--learned_hand_generations", default="1")
    parser.add_argument("--scripted_only_path", default=None)
    parser.add_argument("--single_hand_path", default=None)
    parser.add_argument("--league_path", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--history_length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = os.path.join(args.base_dir, "comparison_tests")

    robots = build_robots(args)
    tests = build_tests(args)
    if not tests:
        raise ValueError("No tests selected.")

    results = []
    for robot_name, robot_path in robots.items():
        if not os.path.exists(robot_path):
            print(f"[Skip] Missing robot {robot_name}: {robot_path}")
            continue
        for test_name, test_type, hand_path in tests:
            if test_type == "learned" and (hand_path is None or not os.path.exists(hand_path)):
                print(f"[Skip] Missing learned hand {test_name}: {hand_path}")
                continue
            print(f"\nEvaluating {robot_name} vs {test_name}")
            if test_type == "mouse":
                metrics = run_mouse_hand_test(
                    robot_name=robot_name,
                    robot_path=robot_path,
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    history_length=args.history_length,
                    seed=args.seed,
                    fps=args.fps,
                )
            else:
                metrics = evaluate_robot_on_test(
                    robot_name=robot_name,
                    robot_path=robot_path,
                    test_name=test_name,
                    test_type=test_type,
                    hand_path=hand_path,
                    episodes=args.episodes,
                    max_steps=args.max_steps,
                    history_length=args.history_length,
                    seed=args.seed,
                )
            results.append(metrics)
            print(
                f"  TIS={metrics.tis_mean:.3f} | ZPD={metrics.zpd_coverage_mean:.3f} | "
                f"Len={metrics.episode_length_mean:.1f} | Catch={metrics.catch_rate:.1%}"
            )

    if not results:
        print("No results generated.")
        return

    save_results(results, args.output)
    print_summary(results)


if __name__ == "__main__":
    main()
