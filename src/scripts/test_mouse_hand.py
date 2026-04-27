"""Mouse-controlled hand test script.

Use mouse to control the hand while a trained robot agent tries to catch it.
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
    """Test trained robot agent with mouse-controlled hand.

    Args:
        model_path: Path to trained robot model (SAC)
        max_steps: Maximum steps per episode
        fps: Target frames per second (default: 60)
    """
    pygame.init()

    grid_size = 10
    cell_size = 50
    width_px = int(grid_size * cell_size * 1.5)
    height_px = int(grid_size * cell_size)

    screen = pygame.display.set_mode((width_px, height_px))
    pygame.display.set_caption("Human vs Robot - You are the HAND! Press Q to quit, R to reset, +/- to adjust FPS")
    clock = pygame.time.Clock()

    current_fps = fps

    # Load robot model
    robot_model = None
    if model_path:
        try:
            robot_model = PPO.load(model_path, verbose=0)
            print(f"Robot model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using random actions for Robot.")

    # Create environment with robot training mode
    env = RehabilitationEnv(
        training_mode='robot',
        robot_model=robot_model,
        hand_model_paths=None
    )

    env.grid_size = grid_size
    env.cell_size = cell_size
    env.random_noise = False


    env.margin = 0.1
    # 开启旁路模式：鼠标直接控制 hand_position，跳过物理约束
    env._bypass_hand_physics = True
    env.distance_threshold_collision = 1.5

    running = True
    episode_count = 0

    while running:
        obs, info = env.reset()
        env.stride_robot = 0.6
        env.stride_hand = 0.3
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
            vec_to_mouse = target_pos - env.hand_position
            dist_to_mouse = np.linalg.norm(vec_to_mouse)

            if dist_to_mouse > 1e-4:
                move_dist = min(dist_to_mouse, env.stride_hand)
                hand_move = (vec_to_mouse / dist_to_mouse) * move_dist
            else:
                hand_move = np.zeros(2)

            # Apply hand movement directly (mouse control bypasses env's hand logic)
            # 同步更新物理惯性状态，防止下一帧惯性滤波叠加残余速度
            env.last_hand_actual_move = hand_move.copy()
            env.hand_position += hand_move
            env.hand_position = np.clip(
                env.hand_position,
                env.margin,
                [env.env_width - env.margin, env.env_height - env.margin]
            )
            env.hand_history_buffer.append(hand_move)

            # Get robot action and step
            obs = env._get_obs()
            if robot_model is not None:
                action, _ = robot_model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

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
    parser = argparse.ArgumentParser(description='Test robot with mouse-controlled hand')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained robot model (PPO)')
    parser.add_argument('--steps', type=int, default=1000,
                        help='Max steps per episode')
    parser.add_argument('--fps', type=int, default=8,
                        help='Target frames per second (default: 60)')

    args = parser.parse_args()

    test_with_mouse(
        model_path=args.model,
        max_steps=args.steps,
        fps=args.fps
    )
