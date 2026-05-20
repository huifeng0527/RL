"""
Trajectory Replay Renderer
===========================
将保存的轨迹数据回放渲染为视频或动画。

用法:
    # 方式1: 生成 mp4 视频
    python replay_renderer.py --input debug_trajectories/episode_20260520_123456.npz

    # 方式2: 生成 gif 动画
    python replay_renderer.py --input debug_trajectories/episode_20260520_123456.npz --output replay.gif

    # 方式3: 交互式播放（matplotlib）
    python replay_renderer.py --input debug_trajectories/episode_20260520_123456.npz --interactive

    # 方式4: 从 JSON 加载
    python replay_renderer.py --input debug_trajectories/episode_20260520_123456.json --json
"""

import os
import sys
import argparse
import numpy as np
import json

# 添加项目根目录
_rl_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _rl_root not in sys.path:
    sys.path.insert(0, _rl_root)

# 导入渲染器
from custom_env.renderer import EnvironmentRenderer


def load_trajectory_data(path: str, from_json: bool = False):
    """加载轨迹数据"""
    if from_json:
        with open(path, "r") as f:
            data = json.load(f)
        return {
            "timestamps": np.array(data["timestamps"]),
            "robot_positions": np.array(data["robot_positions"]),
            "hand_positions": np.array(data["hand_positions"]),
        }
    else:
        npz = np.load(path)
        return {
            "timestamps": npz["timestamps"],
            "robot_positions": npz["robot_positions"],
            "hand_positions": npz["hand_positions"],
        }


def render_replay_mp4(data, output_path: str, grid_size=10, cell_size=50, fps=25):
    """
    使用 pygame + cv2 生成 mp4 视频
    """
    import pygame
    import cv2

    timestamps = data["timestamps"]
    robot_pos = data["robot_positions"]
    hand_pos = data["hand_positions"]

    n_steps = len(timestamps)
    w = int(grid_size * 1.5 * cell_size)
    h = int(grid_size * cell_size)

    pygame.init()
    window = pygame.display.set_mode((w, h))
    clock = pygame.time.Clock()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    font = pygame.font.SysFont("Arial", 18)

    for i in range(n_steps):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                out.release()
                return

        canvas = pygame.Surface((w, h))
        canvas.fill((250, 250, 250))

        scale = cell_size
        to_px = lambda p: (int(p[0] * scale), int(p[1] * scale))

        # 绘制历史轨迹
        for t in range(max(0, i - 50), i):
            alpha_r = (t / i) * 150
            r_px = to_px(robot_pos[t])
            h_px = to_px(hand_pos[t])
            pygame.draw.circle(canvas, (100, 150, 255, alpha_r), r_px, 3)
            pygame.draw.circle(canvas, (255, 100, 100, alpha_r), h_px, 3)

        # 绘制当前位置
        r_px = to_px(robot_pos[i])
        h_px = to_px(hand_pos[i])
        pygame.draw.circle(canvas, (0, 100, 255), r_px, 15)   # Robot 蓝色
        pygame.draw.circle(canvas, (255, 0, 0), h_px, 12)     # Hand 红色

        # 绘制连线（hand -> robot）
        pygame.draw.line(canvas, (0, 200, 0), h_px, r_px, 2)

        # 信息文字
        time_text = font.render(f"t={timestamps[i]:.2f}s  step={i}/{n_steps}", True, (50, 50, 50))
        canvas.blit(time_text, (10, 10))

        # 计算实时距离
        dist = np.linalg.norm(robot_pos[i] - hand_pos[i])
        dist_text = font.render(f"dist={dist:.3f}", True, (100, 100, 100))
        canvas.blit(dist_text, (10, 35))

        window.blit(canvas, (0, 0))
        pygame.display.flip()

        # 转换为 cv2 图像并写入视频
        frame_arr = pygame.surfarray.array3d(canvas)
        frame_arr = np.transpose(frame_arr, (1, 0, 2))
        frame_bgr = cv2.cvtColor(frame_arr, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

        clock.tick(fps)

    out.release()
    pygame.quit()
    print(f"[ReplayRenderer] Video saved: {output_path}")


def render_replay_gif(data, output_path: str, grid_size=10, cell_size=50, skip=5):
    """
    生成 gif 动画（每 skip 帧采样一次）
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    timestamps = data["timestamps"]
    robot_pos = data["robot_positions"]
    hand_pos = data["hand_positions"]

    n_steps = len(timestamps)
    indices = list(range(0, n_steps, skip))

    fig, ax = plt.subplots(figsize=(int(grid_size * 1.5), grid_size))
    ax.set_xlim(0, grid_size * 1.5)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.set_facecolor('#fafafa')

    robot_scatter = ax.scatter([], [], c='blue', s=80, label='Robot', zorder=10)
    hand_scatter = ax.scatter([], [], c='red', s=60, label='Hand', zorder=10)

    traj_robot, = ax.plot([], [], 'b-', alpha=0.3, linewidth=1, label='_nolegend_')
    traj_hand, = ax.plot([], [], 'r-', alpha=0.3, linewidth=1, label='_nolegend_')

    info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', fontfamily='monospace')

    ax.legend(loc='upper right')
    ax.set_title('Trajectory Replay', fontsize=12)

    def init():
        robot_scatter.set_offsets([[], []])
        hand_scatter.set_offsets([[], []])
        traj_robot.set_data([], [])
        traj_hand.set_data([], [])
        info_text.set_text('')
        return robot_scatter, hand_scatter, traj_robot, traj_hand, info_text

    def update(frame_idx):
        i = indices[frame_idx]
        robot_scatter.set_offsets([robot_pos[i]])
        hand_scatter.set_offsets([hand_pos[i]])

        traj_robot.set_data(robot_pos[:i+1, 0], robot_pos[:i+1, 1])
        traj_hand.set_data(hand_pos[:i+1, 0], hand_pos[:i+1, 1])

        dist = np.linalg.norm(robot_pos[i] - hand_pos[i])
        info_text.set_text(f't={timestamps[i]:.2f}s  step={i}  dist={dist:.3f}')
        return robot_scatter, hand_scatter, traj_robot, traj_hand, info_text

    ani = animation.FuncAnimation(fig, update, frames=len(indices),
                                  init_func=init, blit=True, interval=50)
    ani.save(output_path, writer='pillow', fps=20)
    plt.close()
    print(f"[ReplayRenderer] GIF saved: {output_path}")


def interactive_playback(data, grid_size=10, cell_size=50):
    """
    交互式 matplotlib 回放（键盘控制）
    """
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    timestamps = data["timestamps"]
    robot_pos = data["robot_positions"]
    hand_pos = data["hand_positions"]

    n_steps = len(timestamps)
    fig, ax = plt.subplots(figsize=(int(grid_size * 1.5 * 0.8), grid_size * 0.8))
    ax.set_xlim(0, grid_size * 1.5)
    ax.set_ylim(0, grid_size)
    ax.set_aspect('equal')
    ax.set_facecolor('#f5f5f5')
    ax.set_title('Trajectory Replay — Press SPACE to play/pause, Q to quit')
    ax.legend(['Robot', 'Hand'], loc='upper right')

    robot_scatter = ax.scatter([], [], c='blue', s=100, zorder=10)
    hand_scatter = ax.scatter([], [], c='red', s=80, zorder=10)
    line, = ax.plot([], [], 'k-', alpha=0.5, linewidth=1)
    info_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                        fontsize=11, fontfamily='monospace', verticalalignment='top')

    playing = [True]
    current_idx = [0]

    def update_frame():
        i = current_idx[0]
        robot_scatter.set_offsets([robot_pos[i]])
        hand_scatter.set_offsets([hand_pos[i]])
        line.set_data([hand_pos[:i+1, 0], robot_pos[:i+1, 0]],
                      [hand_pos[:i+1, 1], robot_pos[:i+1, 1]])

        dist = np.linalg.norm(robot_pos[i] - hand_pos[i])
        info_text.set_text(f't={timestamps[i]:.2f}s  step={i}/{n_steps}  dist={dist:.3f}')
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == ' ':
            playing[0] = not playing[0]
        elif event.key == 'q':
            plt.close()
        elif event.key == 'right':
            current_idx[0] = min(n_steps - 1, current_idx[0] + 10)
            update_frame()
        elif event.key == 'left':
            current_idx[0] = max(0, current_idx[0] - 10)
            update_frame()

    def animate():
        if playing[0]:
            current_idx[0] = (current_idx[0] + 1) % n_steps
            update_frame()
        fig.canvas.callbacks.CallbackRegistry.started = False  # workaround
        try:
            fig.canvas.get_tk_widget().after(50, animate)
        except Exception:
            plt.close()

    fig.canvas.mpl_connect('key_press_event', on_key)
    update_frame()
    animate()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Trajectory Replay Renderer')
    parser.add_argument('--input', '-i', required=True, help='Path to .npz or .json trajectory file')
    parser.add_argument('--output', '-o', default=None, help='Output path (auto-detect extension)')
    parser.add_argument('--json', action='store_true', help='Load from JSON instead of NPZ')
    parser.add_argument('--fps', type=int, default=25, help='Video FPS')
    parser.add_argument('--grid_size', type=int, default=10, help='Grid size')
    parser.add_argument('--cell_size', type=int, default=50, help='Cell size in pixels')
    parser.add_argument('--interactive', action='store_true', help='Interactive matplotlib playback')
    parser.add_argument('--gif', action='store_true', help='Generate GIF instead of MP4')
    args = parser.parse_args()

    print(f"[ReplayRenderer] Loading: {args.input}")
    data = load_trajectory_data(args.input, from_json=args.json)
    print(f"  -> {len(data['timestamps'])} steps loaded")

    if args.interactive:
        interactive_playback(data, grid_size=args.grid_size, cell_size=args.cell_size)
    elif args.gif or (args.output and args.output.endswith('.gif')):
        output = args.output or args.input.replace('.npz', '_replay.gif').replace('.json', '_replay.gif')
        render_replay_gif(data, output, grid_size=args.grid_size, cell_size=args.cell_size)
    else:
        output = args.output or args.input.replace('.npz', '_replay.mp4').replace('.json', '_replay.mp4')
        render_replay_mp4(data, output, grid_size=args.grid_size, cell_size=args.cell_size, fps=args.fps)


if __name__ == '__main__':
    main()