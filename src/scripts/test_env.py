"""Repeated comparison evaluator for scripted-only, single-H1, and league robots."""

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import pygame
import torch
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.custom_env import RehabilitationEnv
from src.observation_schema import INTERACTION_HISTORY_CHANNELS, model_obs_dim, obs_dim
from src.renderer import render_aesthetic
from src.utils.apf_controller import ReactiveAPFController


DEFAULT_BASE_DIR = r"C:\Users\admin\Desktop\research\RL\logs\league_paper_gru_multistep_aux_pfsp_window_20iter"


@dataclass
class EpisodeRecord:
    trial_index: int
    trial_seed: int
    episode_index: int
    episode_seed: int
    completed: bool
    reward: float
    tiz: float
    zpd_coverage: float
    episode_length: int
    caught: bool
    too_close_rate: float
    too_far_rate: float
    avg_distance: float
    done_reason: str


@dataclass
class TrialMetrics:
    trial_index: int
    trial_seed: int
    requested_episodes: int
    completed_episodes: int
    is_complete: bool
    reward_mean: float | None
    reward_sample_variance: float | None
    reward_sample_std: float | None
    tiz_mean: float | None
    tiz_sample_variance: float | None
    tiz_sample_std: float | None
    zpd_coverage_mean: float | None
    zpd_coverage_sample_variance: float | None
    zpd_coverage_sample_std: float | None
    episode_length_mean: float | None
    episode_length_sample_variance: float | None
    episode_length_sample_std: float | None
    catch_rate: float | None
    too_close_rate: float | None
    too_far_rate: float | None
    avg_distance_mean: float | None
    avg_distance_sample_variance: float | None
    avg_distance_sample_std: float | None
    episodes: list[EpisodeRecord]


@dataclass
class EvaluationSummary:
    robot_name: str
    robot_path: str
    test_name: str
    test_type: str
    hand_path: str | None
    trials_requested: int
    trials_completed: int
    episodes_per_trial: int
    max_steps: int
    base_seed: int
    reward_mean: float | None
    reward_sample_variance: float | None
    reward_sample_std: float | None
    tiz_mean: float | None
    tiz_sample_variance: float | None
    tiz_sample_std: float | None
    zpd_coverage_mean: float | None
    zpd_coverage_sample_variance: float | None
    zpd_coverage_sample_std: float | None
    episode_length_mean: float | None
    episode_length_sample_variance: float | None
    episode_length_sample_std: float | None
    catch_rate: float | None
    too_close_rate: float | None
    too_far_rate: float | None
    avg_distance_mean: float | None
    avg_distance_sample_variance: float | None
    avg_distance_sample_std: float | None
    trials: list[TrialMetrics]


def learned_hand_path(base_dir: str, generation: int):
    return os.path.join(base_dir, f"iteration_{generation}", "hand", "hand", "best_model.zip")


def infer_robot_history_mode(robot_model: PPO | None, history_length: int):
    if robot_model is None:
        return "motion"
    expected_dim = model_obs_dim(robot_model)
    interaction_dim = obs_dim(history_length, 0, INTERACTION_HISTORY_CHANNELS)
    return "interaction" if expected_dim == interaction_dim else "motion"


def make_env(robot_model: PPO | None, hand_model: PPO | None, history_length: int, scripted_prob: float):
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


def mouse_action_from_target(
    hand_position: np.ndarray,
    target_position: np.ndarray,
    stride_hand: float,
) -> np.ndarray:
    stride = max(float(stride_hand), 1e-8)
    action = (target_position - hand_position) / stride
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def sample_stats(values):
    arr = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if arr.size == 0:
        return None, None, None
    mean = float(np.mean(arr))
    if arr.size < 2:
        return mean, None, None
    variance = float(np.var(arr, ddof=1))
    return mean, variance, float(np.sqrt(variance))


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def spawn_seed_plan(base_seed: int, trials: int, episodes_per_trial: int):
    trial_sequences = np.random.SeedSequence(base_seed).spawn(trials)
    plan = []
    for trial_sequence in trial_sequences:
        trial_seed = int(trial_sequence.generate_state(1, dtype=np.uint32)[0])
        episode_sequences = trial_sequence.spawn(episodes_per_trial)
        episode_seeds = [
            int(sequence.generate_state(1, dtype=np.uint32)[0])
            for sequence in episode_sequences
        ]
        plan.append((trial_seed, episode_seeds))
    return plan


def build_episode_record(
    trial_index: int,
    trial_seed: int,
    episode_index: int,
    episode_seed: int,
    total_reward: float,
    distances: list[float],
    done_reason: str,
    zpd_min: float,
    zpd_max: float,
    max_steps: int,
):
    dist_arr = np.asarray(distances, dtype=np.float32)
    episode_length = len(distances)
    in_zpd = (
        (dist_arr >= zpd_min) & (dist_arr <= zpd_max)
        if episode_length
        else np.asarray([], dtype=bool)
    )
    return EpisodeRecord(
        trial_index=trial_index,
        trial_seed=trial_seed,
        episode_index=episode_index,
        episode_seed=episode_seed,
        completed=True,
        reward=float(total_reward),
        tiz=float(np.sum(in_zpd) / max_steps) if episode_length else 0.0,
        zpd_coverage=float(np.mean(in_zpd)) if episode_length else 0.0,
        episode_length=episode_length,
        caught=done_reason == "Robot Caught",
        too_close_rate=float(np.mean(dist_arr < zpd_min)) if episode_length else 0.0,
        too_far_rate=float(np.mean(dist_arr > zpd_max)) if episode_length else 0.0,
        avg_distance=float(np.mean(dist_arr)) if episode_length else 0.0,
        done_reason=done_reason,
    )


def summarize_trial(
    trial_index: int,
    trial_seed: int,
    requested_episodes: int,
    episodes: list[EpisodeRecord],
):
    reward_mean, reward_variance, reward_std = sample_stats([item.reward for item in episodes])
    tiz_mean, tiz_variance, tiz_std = sample_stats([item.tiz for item in episodes])
    zpd_mean, zpd_variance, zpd_std = sample_stats([item.zpd_coverage for item in episodes])
    length_mean, length_variance, length_std = sample_stats([item.episode_length for item in episodes])
    distance_mean, distance_variance, distance_std = sample_stats([item.avg_distance for item in episodes])
    return TrialMetrics(
        trial_index=trial_index,
        trial_seed=trial_seed,
        requested_episodes=requested_episodes,
        completed_episodes=len(episodes),
        is_complete=len(episodes) == requested_episodes,
        reward_mean=reward_mean,
        reward_sample_variance=reward_variance,
        reward_sample_std=reward_std,
        tiz_mean=tiz_mean,
        tiz_sample_variance=tiz_variance,
        tiz_sample_std=tiz_std,
        zpd_coverage_mean=zpd_mean,
        zpd_coverage_sample_variance=zpd_variance,
        zpd_coverage_sample_std=zpd_std,
        episode_length_mean=length_mean,
        episode_length_sample_variance=length_variance,
        episode_length_sample_std=length_std,
        catch_rate=float(np.mean([item.caught for item in episodes])) if episodes else None,
        too_close_rate=float(np.mean([item.too_close_rate for item in episodes])) if episodes else None,
        too_far_rate=float(np.mean([item.too_far_rate for item in episodes])) if episodes else None,
        avg_distance_mean=distance_mean,
        avg_distance_sample_variance=distance_variance,
        avg_distance_sample_std=distance_std,
        episodes=episodes,
    )


def summarize_evaluation(
    robot_name: str,
    robot_path: str,
    test_name: str,
    test_type: str,
    hand_path: str | None,
    trials_requested: int,
    episodes_per_trial: int,
    max_steps: int,
    base_seed: int,
    trials: list[TrialMetrics],
):
    completed = [trial for trial in trials if trial.is_complete]
    reward_mean, reward_variance, reward_std = sample_stats([trial.reward_mean for trial in completed])
    tiz_mean, tiz_variance, tiz_std = sample_stats([trial.tiz_mean for trial in completed])
    zpd_mean, zpd_variance, zpd_std = sample_stats([trial.zpd_coverage_mean for trial in completed])
    length_mean, length_variance, length_std = sample_stats([trial.episode_length_mean for trial in completed])
    distance_mean, distance_variance, distance_std = sample_stats([trial.avg_distance_mean for trial in completed])
    return EvaluationSummary(
        robot_name=robot_name,
        robot_path=robot_path,
        test_name=test_name,
        test_type=test_type,
        hand_path=hand_path,
        trials_requested=trials_requested,
        trials_completed=len(completed),
        episodes_per_trial=episodes_per_trial,
        max_steps=max_steps,
        base_seed=base_seed,
        reward_mean=reward_mean,
        reward_sample_variance=reward_variance,
        reward_sample_std=reward_std,
        tiz_mean=tiz_mean,
        tiz_sample_variance=tiz_variance,
        tiz_sample_std=tiz_std,
        zpd_coverage_mean=zpd_mean,
        zpd_coverage_sample_variance=zpd_variance,
        zpd_coverage_sample_std=zpd_std,
        episode_length_mean=length_mean,
        episode_length_sample_variance=length_variance,
        episode_length_sample_std=length_std,
        catch_rate=float(np.mean([trial.catch_rate for trial in completed])) if completed else None,
        too_close_rate=float(np.mean([trial.too_close_rate for trial in completed])) if completed else None,
        too_far_rate=float(np.mean([trial.too_far_rate for trial in completed])) if completed else None,
        avg_distance_mean=distance_mean,
        avg_distance_sample_variance=distance_variance,
        avg_distance_sample_std=distance_std,
        trials=trials,
    )


def run_automatic_trial(
    robot_model: PPO,
    hand_model: PPO | None,
    test_type: str,
    history_length: int,
    max_steps: int,
    trial_index: int,
    trial_seed: int,
    episode_seeds: list[int],
):
    env = make_env(
        robot_model=robot_model,
        hand_model=hand_model,
        history_length=history_length,
        scripted_prob=0.0 if test_type == "learned" else 1.0,
    )
    env.max_steps = max_steps
    episodes = []
    try:
        for episode_index, episode_seed in enumerate(episode_seeds):
            seed_everything(episode_seed)
            obs, _ = env.reset(seed=episode_seed)
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

            episodes.append(
                build_episode_record(
                    trial_index=trial_index,
                    trial_seed=trial_seed,
                    episode_index=episode_index,
                    episode_seed=episode_seed,
                    total_reward=total_reward,
                    distances=distances,
                    done_reason=done_reason,
                    zpd_min=float(env.zpd_min),
                    zpd_max=float(env.zpd_max),
                    max_steps=max_steps,
                )
            )
    finally:
        env.close()

    return summarize_trial(trial_index, trial_seed, len(episode_seeds), episodes)


def evaluate_robot_on_test(
    robot_name: str,
    robot_path: str,
    test_name: str,
    test_type: str,
    hand_path: str | None,
    trials: int,
    episodes_per_trial: int,
    max_steps: int,
    history_length: int,
    base_seed: int,
):
    robot_model = PPO.load(robot_path, verbose=0)
    hand_model = None
    if test_type == "learned":
        hand_model = PPO.load(
            hand_path,
            custom_objects={"learning_rate": 0.0, "optimizer_class": None},
            verbose=0,
        )

    trial_results = []
    for trial_index, (trial_seed, episode_seeds) in enumerate(
        spawn_seed_plan(base_seed, trials, episodes_per_trial)
    ):
        trial_result = run_automatic_trial(
            robot_model=robot_model,
            hand_model=hand_model,
            test_type=test_type,
            history_length=history_length,
            max_steps=max_steps,
            trial_index=trial_index,
            trial_seed=trial_seed,
            episode_seeds=episode_seeds,
        )
        trial_results.append(trial_result)
        print(
            f"  Trial {trial_index + 1}/{trials}: "
            f"TIZ={trial_result.tiz_mean:.3f} | Len={trial_result.episode_length_mean:.1f}"
        )

    return summarize_evaluation(
        robot_name=robot_name,
        robot_path=robot_path,
        test_name=test_name,
        test_type=test_type,
        hand_path=hand_path,
        trials_requested=trials,
        episodes_per_trial=episodes_per_trial,
        max_steps=max_steps,
        base_seed=base_seed,
        trials=trial_results,
    )


def run_mouse_trial(
    robot_name: str,
    robot_model: PPO | None,
    apf_controller: ReactiveAPFController | None,
    history_length: int,
    max_steps: int,
    fps: int,
    trial_index: int,
    trial_seed: int,
    episode_seeds: list[int],
    screen,
    clock,
    font,
):
    env = make_env(
        robot_model=robot_model,
        hand_model=None,
        history_length=history_length,
        scripted_prob=1.0,
    )
    env.max_steps = max_steps
    env.random_noise = False

    current_mouse_action = {"value": np.zeros(2, dtype=np.float32)}
    original_resolve_hand_move = env._resolve_hand_move

    def resolve_mouse_move(_action):
        hand_intent = current_mouse_action["value"] * env.stride_hand
        return env._apply_hand_execution(hand_intent)

    env._resolve_hand_move = resolve_mouse_move
    episodes = []
    quit_requested = False

    try:
        for episode_index, episode_seed in enumerate(episode_seeds):
            while True:
                seed_everything(episode_seed)
                obs, _ = env.reset(seed=episode_seed)
                current_mouse_action["value"] = np.zeros(2, dtype=np.float32)
                if apf_controller is not None:
                    apf_controller.reset()
                pygame.display.set_caption(
                    f"Mouse hand test: {robot_name} | Trial {trial_index + 1} | "
                    f"Episode {episode_index + 1}/{len(episode_seeds)}"
                )

                terminated = False
                truncated = False
                restart_requested = False
                total_reward = 0.0
                distances = []
                mouse_action_magnitudes = []
                hand_step_magnitudes = []
                done_reason = ""

                print(
                    f"\nMouse trial {trial_index + 1}, episode {episode_index + 1}/{len(episode_seeds)}. "
                    f"stride_hand={env.stride_hand:.3f}, stride_robot={env.stride_robot:.3f}, "
                    f"collision_threshold={env.distance_threshold_collision:.3f}, fps={fps}. "
                    "Press Q to quit or R to restart the episode."
                )

                while not (terminated or truncated or quit_requested or restart_requested):
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            quit_requested = True
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_q:
                                quit_requested = True
                            elif event.key == pygame.K_r:
                                restart_requested = True

                    if quit_requested or restart_requested:
                        continue

                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    target = np.array([
                        np.clip(mouse_x / env.cell_size, env.margin, env.env_width - env.margin),
                        np.clip(mouse_y / env.cell_size, env.margin, env.env_height - env.margin),
                    ], dtype=np.float32)
                    current_mouse_action["value"] = mouse_action_from_target(
                        hand_position=np.asarray(env.hand_position, dtype=np.float32),
                        target_position=target,
                        stride_hand=float(env.stride_hand),
                    )
                    mouse_action_magnitudes.append(
                        float(np.linalg.norm(current_mouse_action["value"]))
                    )

                    old_hand_position = np.asarray(env.hand_position, dtype=np.float32).copy()
                    if apf_controller is not None:
                        action = apf_controller.predict(env)
                    elif robot_model is not None:
                        action, _ = robot_model.predict(obs, deterministic=True)
                    else:
                        raise RuntimeError("No robot controller is available for mouse evaluation.")
                    obs, reward, terminated, truncated, info = env.step(action)
                    actual_hand_move = (
                        np.asarray(env.hand_position, dtype=np.float32)
                        - old_hand_position
                    )
                    hand_step_magnitudes.append(float(np.linalg.norm(actual_hand_move)))
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
                        f"{robot_name} | trial {trial_index + 1} | "
                        f"ep {episode_index + 1}/{len(episode_seeds)} | "
                        f"dist {info.get('dist', 0):.2f} | step {len(distances)}/{max_steps}",
                        True,
                        (0, 0, 0),
                    )
                    screen.blit(text, (10, 10))
                    pygame.display.flip()
                    clock.tick(fps)

                if quit_requested:
                    break
                if restart_requested:
                    print("  Episode restarted; the same episode seed will be reused.")
                    continue

                if hand_step_magnitudes:
                    mouse_actions = np.asarray(
                        mouse_action_magnitudes,
                        dtype=np.float64,
                    )
                    hand_steps = np.asarray(hand_step_magnitudes, dtype=np.float64)
                    print(
                        "  Hand-step debug: "
                        f"axis_max={env.stride_hand:.3f}, "
                        f"l2_max={np.sqrt(2.0) * env.stride_hand:.3f}, "
                        f"action_norm_mean={mouse_actions.mean():.3f}, "
                        f"actual_mean={hand_steps.mean():.3f}, "
                        f"actual_std={hand_steps.std():.3f}, "
                        f"actual_min={hand_steps.min():.3f}, "
                        f"actual_max={hand_steps.max():.3f}, "
                        f"mean_rate@{fps}Hz={hand_steps.mean() * fps:.3f}/s"
                    )

                episodes.append(
                    build_episode_record(
                        trial_index=trial_index,
                        trial_seed=trial_seed,
                        episode_index=episode_index,
                        episode_seed=episode_seed,
                        total_reward=total_reward,
                        distances=distances,
                        done_reason=done_reason,
                        zpd_min=float(env.zpd_min),
                        zpd_max=float(env.zpd_max),
                        max_steps=max_steps,
                    )
                )
                break

            if quit_requested:
                break
    finally:
        env._resolve_hand_move = original_resolve_hand_move
        env.close()

    return summarize_trial(trial_index, trial_seed, len(episode_seeds), episodes), quit_requested


def wait_for_mouse_trial(screen, clock, font, robot_name: str, trial_index: int, trials: int):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    return False
                if event.key in {pygame.K_RETURN, pygame.K_SPACE}:
                    return True

        screen.fill((245, 247, 250))
        lines = [
            f"{robot_name}: mouse trial {trial_index + 1}/{trials}",
            "Press Enter or Space to start; press Q to quit.",
        ]
        for line_index, line in enumerate(lines):
            text = font.render(line, True, (0, 0, 0))
            screen.blit(text, (30, 210 + line_index * 32))
        pygame.display.flip()
        clock.tick(30)


def run_mouse_hand_test(
    robot_name: str,
    robot_path: str | None,
    trials: int,
    episodes_per_trial: int,
    max_steps: int,
    history_length: int,
    base_seed: int,
    fps: int,
):
    if robot_name == "reactive_apf":
        robot_model = None
        apf_controller = ReactiveAPFController()
        controller_path = "ReactiveAPFController(radial+tangent+boundary)"
    else:
        robot_model = PPO.load(robot_path, verbose=0)
        apf_controller = None
        controller_path = str(robot_path)

    pygame.init()
    screen = pygame.display.set_mode((750, 500))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 18)
    trial_results = []
    quit_requested = False

    try:
        for trial_index, (trial_seed, episode_seeds) in enumerate(
            spawn_seed_plan(base_seed, trials, episodes_per_trial)
        ):
            if not wait_for_mouse_trial(screen, clock, font, robot_name, trial_index, trials):
                quit_requested = True
                break
            print(
                f"\nStarting mouse trial {trial_index + 1}/{trials} for {robot_name} "
                f"with {episodes_per_trial} episodes."
            )
            trial_result, quit_requested = run_mouse_trial(
                robot_name=robot_name,
                robot_model=robot_model,
                apf_controller=apf_controller,
                history_length=history_length,
                max_steps=max_steps,
                fps=fps,
                trial_index=trial_index,
                trial_seed=trial_seed,
                episode_seeds=episode_seeds,
                screen=screen,
                clock=clock,
                font=font,
            )
            trial_results.append(trial_result)
            status = "complete" if trial_result.is_complete else "incomplete"
            print(f"  Mouse trial {trial_index + 1}: {status}")
            if quit_requested:
                break
    finally:
        pygame.quit()

    summary = summarize_evaluation(
        robot_name=robot_name,
        robot_path=controller_path,
        test_name="mouse_hand",
        test_type="mouse",
        hand_path=None,
        trials_requested=trials,
        episodes_per_trial=episodes_per_trial,
        max_steps=max_steps,
        base_seed=base_seed,
        trials=trial_results,
    )
    return summary, quit_requested


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
        for generation in generations:
            path = learned_hand_path(args.base_dir, generation)
            tests.append((f"agent_H{generation}", "learned", path))
    if args.test_set == "mouse":
        tests.append(("mouse_hand", "mouse", None))
    return tests


def default_robot_paths(base_dir: str):
    return {
        "scripted_only": os.path.join(base_dir, "baselines", "scripted_only_5m", "robot", "best_model.zip"),
        "single_h1": os.path.join(base_dir, "baselines", "single_hand_h1_5m", "robot", "best_model.zip"),
        "league": os.path.join(base_dir, "iteration_10", "robot", "robot", "best_model.zip"),
    }


def build_robots(args):
    defaults = default_robot_paths(args.base_dir)
    candidates = {
        "scripted_only": args.scripted_only_path or defaults["scripted_only"],
        "single_h1": args.single_hand_path or defaults["single_h1"],
        "league": args.league_path or defaults["league"],
    }
    if args.test_set == "mouse":
        candidates = {"reactive_apf": None, **candidates}
    elif args.robot == "reactive_apf":
        raise ValueError("reactive_apf is currently supported only with --test_set mouse.")
    return candidates if args.robot == "all" else {args.robot: candidates[args.robot]}


def summary_row(result: EvaluationSummary):
    row = asdict(result)
    row.pop("trials")
    return row


def trial_rows(results: list[EvaluationSummary]):
    rows = []
    for result in results:
        metadata = {
            "robot_name": result.robot_name,
            "robot_path": result.robot_path,
            "test_name": result.test_name,
            "test_type": result.test_type,
            "hand_path": result.hand_path,
            "base_seed": result.base_seed,
            "max_steps": result.max_steps,
        }
        for trial in result.trials:
            row = asdict(trial)
            row.pop("episodes")
            rows.append({**metadata, **row})
    return rows


def episode_rows(results: list[EvaluationSummary]):
    rows = []
    for result in results:
        metadata = {
            "robot_name": result.robot_name,
            "robot_path": result.robot_path,
            "test_name": result.test_name,
            "test_type": result.test_type,
            "hand_path": result.hand_path,
            "base_seed": result.base_seed,
            "max_steps": result.max_steps,
        }
        for trial in result.trials:
            for episode in trial.episodes:
                rows.append({**metadata, **asdict(episode)})
    return rows


def write_csv(path: str, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_results(results: list[EvaluationSummary], output_dir: str, configuration: dict):
    os.makedirs(output_dir, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"comparison_results_{run_id}.json")
    summary_path = os.path.join(output_dir, f"comparison_results_{run_id}.csv")
    trials_path = os.path.join(output_dir, f"comparison_trials_{run_id}.csv")
    episodes_path = os.path.join(output_dir, f"comparison_episodes_{run_id}.csv")

    payload = {
        "schema_version": 2,
        "run_id": run_id,
        "configuration": configuration,
        "evaluations": [asdict(result) for result in results],
    }
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    write_csv(summary_path, [summary_row(result) for result in results])
    write_csv(trials_path, trial_rows(results))
    write_csv(episodes_path, episode_rows(results))

    print(f"\nSaved JSON:    {json_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved trials:  {trials_path}")
    print(f"Saved episodes:{episodes_path}")
    return {
        "json": json_path,
        "summary": summary_path,
        "trials": trials_path,
        "episodes": episodes_path,
    }


def format_mean_std(mean: float | None, std: float | None, digits: int = 3):
    if mean is None:
        return "N/A"
    if std is None:
        return f"{mean:.{digits}f} +/- N/A"
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def print_summary(results: list[EvaluationSummary]):
    print("\n" + "=" * 100)
    print("FINAL POLICY COMPARISON: MEAN +/- SAMPLE SD ACROSS COMPLETE TRIAL MEANS")
    print("=" * 100)
    for result in results:
        print(f"{result.robot_name} vs {result.test_name}")
        print(
            f"  TIZ:    {format_mean_std(result.tiz_mean, result.tiz_sample_std)} "
            f"(variance={result.tiz_sample_variance})"
        )
        print(
            f"  Length: {format_mean_std(result.episode_length_mean, result.episode_length_sample_std, 1)} "
            f"(variance={result.episode_length_sample_variance})"
        )
        print(
            f"  Complete trials: {result.trials_completed}/{result.trials_requested}; "
            f"episodes per trial: {result.episodes_per_trial}"
        )
    print("=" * 100)
    print("Episodes estimate each trial mean and are not treated as independent trial replications.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate robot policies over repeated independent trials."
    )
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR)
    parser.add_argument(
        "--robot",
        choices=["reactive_apf", "scripted_only", "single_h1", "league", "all"],
        default="all",
    )
    parser.add_argument("--test_set", choices=["scripted", "agent", "learned", "mouse", "all"], default="all")
    parser.add_argument("--learned_hand_generations", default="1")
    parser.add_argument("--scripted_only_path", default=None)
    parser.add_argument("--single_hand_path", default=None)
    parser.add_argument("--league_path", default=None)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--episodes_per_trial", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--history_length", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.trials < 1:
        raise ValueError("--trials must be at least 1.")
    if args.episodes_per_trial < 1:
        raise ValueError("--episodes_per_trial must be at least 1.")
    if args.output is None:
        args.output = os.path.join(args.base_dir, "comparison_tests")

    robots = build_robots(args)
    tests = build_tests(args)
    if not tests:
        raise ValueError("No tests selected.")

    results = []
    stop_requested = False
    for robot_name, robot_path in robots.items():
        if robot_name != "reactive_apf" and not os.path.exists(robot_path):
            print(f"[Skip] Missing robot {robot_name}: {robot_path}")
            continue
        for test_name, test_type, hand_path in tests:
            if test_type == "learned" and (hand_path is None or not os.path.exists(hand_path)):
                print(f"[Skip] Missing learned hand {test_name}: {hand_path}")
                continue
            print(f"\nEvaluating {robot_name} vs {test_name}")
            if test_type == "mouse":
                metrics, stop_requested = run_mouse_hand_test(
                    robot_name=robot_name,
                    robot_path=robot_path,
                    trials=args.trials,
                    episodes_per_trial=args.episodes_per_trial,
                    max_steps=args.max_steps,
                    history_length=args.history_length,
                    base_seed=args.seed,
                    fps=args.fps,
                )
            else:
                metrics = evaluate_robot_on_test(
                    robot_name=robot_name,
                    robot_path=robot_path,
                    test_name=test_name,
                    test_type=test_type,
                    hand_path=hand_path,
                    trials=args.trials,
                    episodes_per_trial=args.episodes_per_trial,
                    max_steps=args.max_steps,
                    history_length=args.history_length,
                    base_seed=args.seed,
                )
            results.append(metrics)
            print(
                f"  Across trials: TIZ={format_mean_std(metrics.tiz_mean, metrics.tiz_sample_std)} | "
                f"Len={format_mean_std(metrics.episode_length_mean, metrics.episode_length_sample_std, 1)}"
            )
            if stop_requested:
                break
        if stop_requested:
            break

    if not results:
        print("No results generated.")
        return

    configuration = {
        "base_dir": os.path.abspath(args.base_dir),
        "base_seed": args.seed,
        "seed_strategy": "numpy.SeedSequence.spawn with Python, NumPy, PyTorch, and Gym reset seeding",
        "trials_requested": args.trials,
        "episodes_per_trial": args.episodes_per_trial,
        "max_steps": args.max_steps,
        "history_length": args.history_length,
        "learned_hand_generations": parse_generations(args.learned_hand_generations),
        "robot_predict_deterministic": True,
        "learned_hand_predict_deterministic": False,
        "tiz_definition": "steps_in_zpd / max_steps",
        "sample_variance_ddof": 1,
    }
    if args.test_set == "mouse":
        configuration.update(
            {
                "mouse_robot_order": list(robots.keys()),
                "mouse_action_mapping": "component-wise action clipping to [-1, 1], then shared hand execution physics",
                "mouse_collision_threshold": 2.0,
                "mouse_stride_hand_range": [0.3, 0.6],
                "mouse_stride_robot_range": [0.58, 0.62],
                "mouse_settings_shared_across_robot_controllers": True,
            }
        )
    save_results(results, args.output, configuration)
    print_summary(results)


if __name__ == "__main__":
    main()
