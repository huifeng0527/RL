"""Mouse-controlled robot test script.

Use mouse to control the robot while a trained hand agent tries to avoid it.
Press 'R' to reset episode, 'Q' to quit, '+'/'-' to adjust FPS.
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


def test_with_mouse(model_path=None, max_steps=1000, fps=60):
    """Test trained hand agent with mouse-controlled robot.

    Args:
        model_path: Path to trained hand model (PPO)
        max_steps: Maximum steps per episode
        fps: Target frames per second (default: 60)
    """
    pygame.init()

    grid_size = 10
    cell_size = 50
    width_px = int(grid_size * cell_size * 1.5)
    height_px = int(grid_size * cell_size)

    screen = pygame.display.set_mode((width_px, height_px))
    pygame.display.set_caption("Human vs Robot - You are the ROBOT! Press Q to quit, R to reset, +/- to adjust FPS")
    clock = pygame.time.Clock()

    current_fps = fps

    # Load hand model
    hand_model = None
    if model_path:
        try:
            hand_model = PPO.load(model_path)
            print(f"Hand model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using scripted hand behavior.")

    # Create environment with ROBOT training mode (so hand_model controls the hand)
    env = RehabilitationEnv(
        training_mode='robot',
        robot_model=None,
        hand_model_paths=None
    )

    if hand_model is not None:
        env.hand_model = hand_model

    env.grid_size = grid_size
    env.cell_size = cell_size
    env.random_noise = False
    env.stride_robot = 0.5
    env.stride_hand = 0.4
    env.margin = 0.1
    env.distance_threshold_collision = 2

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
            # Handle pygame events (for quit and reset)
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

            # Get mouse position and convert to env coordinates
            mouse_x_px, mouse_y_px = pygame.mouse.get_pos()
            target_x = mouse_x_px / cell_size
            target_y = mouse_y_px / cell_size

            # Clip to valid range
            target_x = np.clip(target_x, env.margin, env.env_width - env.margin)
            target_y = np.clip(target_y, env.margin, env.env_height - env.margin)
            target_pos = np.array([target_x, target_y])

            # Calculate smooth movement towards mouse (respect stride limit)
            vec_to_mouse = target_pos - env.robot_position
            dist_to_mouse = np.linalg.norm(vec_to_mouse)

            if dist_to_mouse > 1e-4:
                move_dist = min(dist_to_mouse, env.stride_robot)
                robot_move = (vec_to_mouse / dist_to_mouse) * move_dist
            else:
                robot_move = np.zeros(2)

            # Apply robot movement directly
            env.robot_position += robot_move
            env.robot_position = np.clip(
                env.robot_position,
                env.margin,
                [env.env_width - env.margin, env.env_height - env.margin]
            )
            env.robot_history_buffer.append(robot_move)

            # Temporarily disable env's robot movement
            temp_stride = env.stride_robot
            env.stride_robot = 0.0

            # Step environment - hand will be controlled by hand_model
            action = np.zeros(2)  # robot action doesn't matter since we disabled stride
            obs, reward, terminated, truncated, info = env.step(action)

            # Restore stride
            env.stride_robot = temp_stride

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
    parser = argparse.ArgumentParser(description='Test hand with mouse-controlled robot')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained hand model (PPO)')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Max steps per episode')
    parser.add_argument('--fps', type=int, default=10,
                        help='Target frames per second (default: 60)')

    args = parser.parse_args()

    test_with_mouse(
        model_path=args.model,
        max_steps=args.steps,
        fps=args.fps
    )
