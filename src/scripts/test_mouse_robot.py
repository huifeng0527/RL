"""Mouse-controlled robot test script.

Use mouse to control the robot while a trained hand agent tries to avoid it.
Press 'R' to reset episode, 'Q' to quit.
"""

import sys
import os

# Add project root to path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import pygame
import numpy as np

from src.custom_env import RehabilitationEnv
from src.renderer import render_aesthetic
from stable_baselines3 import PPO


def test_with_mouse(model_path=None, render_mode="human", max_steps=10000):
    """Test trained hand agent with mouse-controlled robot.

    Args:
        model_path: Path to trained hand model. If None, uses scripted hand.
        render_mode: Pygame render mode
        max_steps: Maximum steps per episode
    """
    pygame.init()

    grid_size = 10
    cell_size = 50
    width_px = int(grid_size * cell_size * 1.5)
    height_px = int(grid_size * cell_size)

    screen = pygame.display.set_mode((width_px, height_px))
    pygame.display.set_caption("Mouse-Controlled Robot Test - Press Q to quit, R to reset")

    # Load model if provided
    hand_model = None
    if model_path:
        hand_model = PPO.load(model_path)
        print(f"Loaded hand model from {model_path}")

    # Create environment with hand training mode (hand is the RL agent)
    env = RehabilitationEnv(
        training_mode='hand',
        robot_model=None,
        hand_model_paths=None
    )

    # Override hand_model to use the loaded one
    if hand_model is not None:
        env.hand_model = hand_model

    env.grid_size = grid_size
    env.cell_size = cell_size
    env.random_noise = False  # Disable noise for cleaner testing

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

        while running and not episode_done:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        episode_done = True

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Update robot position to mouse location
                    mouse_x, mouse_y = event.pos
                    env.robot_position = np.array([
                        mouse_x / cell_size,
                        mouse_y / cell_size
                    ])
                    env.robot_position = np.clip(
                        env.robot_position,
                        env.margin,
                        [env.env_width - env.margin, env.env_height - env.margin]
                    )

            # In hand training mode, step with zero action
            # (robot is mouse-controlled, hand uses its policy)
            action = np.zeros(2)  # robot action doesn't matter when mouse-controlled
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

            # Check done
            if terminated or truncated:
                episode_done = True
                print(f"Episode {episode_count} finished:")
                print(f"  Steps: {steps}, Reward: {total_reward:.2f}")
                print(f"  Done reason: {info.get('done_reason', 'unknown')}")

        pygame.time.wait(500)  # Brief pause between episodes

    env.close()
    pygame.quit()
    print("\nTest finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test hand with mouse-controlled robot')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained hand model (PPO/SAC)')
    parser.add_argument('--steps', type=int, default=10000,
                        help='Max steps per episode')

    args = parser.parse_args()

    test_with_mouse(
        model_path=args.model,
        max_steps=args.steps
    )
