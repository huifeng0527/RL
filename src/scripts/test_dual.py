"""Basic dual agent test script.

Loads robot and hand models, runs them against each other with rendering.
Press 'R' to reset episode, 'Q' to quit, '+'/'-' to adjust FPS.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import pygame
import numpy as np

from src.custom_env import RehabilitationEnv
from src.renderer import render_aesthetic
from stable_baselines3 import PPO


def test_dual(robot_model_path=None, hand_model_path=None, max_steps=1000, fps=30):
    """Test both robot and hand agents together.

    Args:
        robot_model_path: Path to trained robot model (PPO)
        hand_model_path: Path to trained hand model (PPO)
        max_steps: Maximum steps per episode
        fps: Target frames per second (default: 30)
    """
    pygame.init()

    grid_size = 10
    cell_size = 50
    width_px = int(grid_size * cell_size * 1.5)
    height_px = int(grid_size * cell_size)

    screen = pygame.display.set_mode((width_px, height_px))
    pygame.display.set_caption("Robot vs Hand - Press Q to quit, R to reset, +/- to adjust FPS")
    clock = pygame.time.Clock()

    current_fps = fps

    # Load models
    robot_model = None
    hand_model = None

    if robot_model_path:
        try:
            robot_model = PPO.load(robot_model_path, verbose=0)
            print(f"Robot model loaded from {robot_model_path}")
        except Exception as e:
            print(f"Error loading robot model: {e}")

    if hand_model_path:
        try:
            hand_model = PPO.load(hand_model_path, verbose=0)
            print(f"Hand model loaded from {hand_model_path}")
        except Exception as e:
            print(f"Error loading hand model: {e}")

    # Create environment - use robot training mode so hand_model controls hand
    env = RehabilitationEnv(
        training_mode='robot',
        robot_model=robot_model,
        hand_model=hand_model
    )

    # if hand_model is not None:
    #     env.hand_model = hand_model

    env.grid_size = grid_size
    env.cell_size = cell_size
    env.random_noise = False

    running = True
    episode_count = 0

    while running:
        obs, info = env.reset()
        episode_count += 1
        steps = 0
        total_reward = 0

        print(f"\nEpisode {episode_count} started")
        print(f"Robot: {env.robot_position}, Hand: {env.hand_position}")

        episode_done = False

        while running and not episode_done and steps < max_steps:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        episode_done = True
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        current_fps = min(current_fps + 10, 300)
                        print(f"FPS: {current_fps}")
                    elif event.key == pygame.K_MINUS:
                        current_fps = max(current_fps - 10, 10)
                        print(f"FPS: {current_fps}")

            # Get actions
            if robot_model is not None:
                action, _ = robot_model.predict(obs, deterministic=True)
            else:
                action = np.zeros(2)

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # Render
            render_aesthetic(
                env.robot_position,
                env.hand_position,
                env.fixed_point,
                env.trajectory_points,
                grid_size=grid_size,
                cell_size=cell_size,
                window=screen
            )

            clock.tick(current_fps)

            # Check done
            if terminated or truncated:
                episode_done = True
                print(f"Episode {episode_count} finished:")
                print(f"  Steps: {steps}, Reward: {total_reward:.2f}")
                print(f"  Done reason: {info.get('done_reason', 'unknown')}")

        if steps >= max_steps:
            print(f"Episode {episode_count} reached max steps ({max_steps})")

    env.close()
    pygame.quit()
    print("\nTest finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test robot vs hand agents')
    parser.add_argument('--robot', type=str, default=r"C:\Users\admin\Desktop\科研\RL\logs\dual_iterative_0423_1539\iteration_2\robot\robot\best_model.zip",
                        help='Path to robot model (PPO)')
    parser.add_argument('--hand', type=str, default="",
                        help='Path to hand model (PPO)')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Max steps per episode')
    parser.add_argument('--fps', type=int, default=8,
                        help='Target frames per second (default: 30)')

    args = parser.parse_args()

    if not args.robot and not args.hand:
        print("Warning: No models provided, using random actions")

    test_dual(
        robot_model_path=args.robot,
        hand_model_path=args.hand,
        max_steps=args.steps,
        fps=args.fps
    )
