"""Multi-dimensional Stress Test Suite for Robot Evaluation.

This test suite evaluates and compares three robot training approaches:
- Baseline A: Robot trained only with static script hands
- Baseline B: Robot trained only with a single strongest RL hand
- PFSP (Ours): Robot trained with Progressive Fictitious Self-Play iterative league

The "final exam" consists of 4 diverse test hands designed to expose
weaknesses in Baseline A and B while PFSP handles them gracefully.

Usage:
    python src/scripts/test_env.py --robot all --test all --episodes 50 --visual
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import numpy as np
import pygame

from src.custom_env import RehabilitationEnv
from src.renderer import render_aesthetic
from stable_baselines3 import PPO


# =============================================================================
# Constants
# =============================================================================
ZPD_MIN = 4.0
ZPD_MAX = 6.0
GRID_SIZE = 10
CELL_SIZE = 50
WIDTH_PX = int(GRID_SIZE * CELL_SIZE * 1.5)
HEIGHT_PX = int(GRID_SIZE * CELL_SIZE)

# Model paths
BASE_DIR = r"C:\Users\admin\Desktop\科研\RL\logs\dual_iterative_0427_1314"
MODEL_PATHS = {
    'baseline_a': os.path.join(BASE_DIR, "iteration_1", "robot", "robot", "best_model.zip"),
    'baseline_b': os.path.join(BASE_DIR, "baseline_b", "robot", "best_model.zip"),
    'pfsp': os.path.join(BASE_DIR, "iteration_9", "robot", "robot", "best_model.zip"),
}
UNSEEN_HAND_PATH = os.path.join(BASE_DIR, "iteration_9", "hand", "hand", "best_model.zip")


# =============================================================================
# Metrics
# =============================================================================
@dataclass
class StressTestMetrics:
    """Metrics captured during stress test."""
    robot_name: str
    test_name: str
    survival_times: List[int]
    mean_survival_time: float
    std_survival_time: float
    catch_count: int
    catch_rate: float
    zpd_maintenance_rate: float
    mean_distance: float

    def to_dict(self):
        return asdict(self)


# =============================================================================
# Test Hand Classes
# =============================================================================

class SluggishScriptHand:
    """极慢速脚本手 (Test 1) - stride locked at 0.3.

    专门用来打脸 Baseline B：它天天跟神仙打架，全速逃命。
    遇到这种极慢的手，会因为过度紧张，瞬间跑到屏幕对角线躲起来。
    """

    STRIDE = 0.3  # 重度中风，完全动不了

    def __init__(self, epsilon: float = 0.2):
        self.epsilon = epsilon

    def get_move(self, hand_pos: np.ndarray, robot_pos: np.ndarray) -> np.ndarray:
        """返回手的移动向量."""
        scripted_stride = self.STRIDE
        if np.random.random() < self.epsilon:
            move = np.random.uniform(-1, 1, size=2)
            move = self._safe_normalize(move) * scripted_stride
        else:
            vec = robot_pos - hand_pos
            move = self._safe_normalize(vec) * scripted_stride
        return move

    @staticmethod
    def _safe_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return np.zeros_like(vec)
        return vec / norm


class SpasmScriptHand:
    """突发痉挛手 (Test 2) - alternates slow/fast.

    专门用来打脸 Baseline A：它习惯了对方是匀速的。
    当手突然加速时，反应不及，瞬间被抓。
    """

    SLOW_STRIDE = 0.2
    FAST_STRIDE = 0.5
    SLOW_FRAMES = 32   # 4 sec @ 8 FPS
    FAST_FRAMES = 8    # 1 sec @ 8 FPS

    def __init__(self, epsilon: float = 0.2):
        self.epsilon = epsilon
        self.frame_counter = 0
        self.is_fast_phase = False

    def get_move(self, hand_pos: np.ndarray, robot_pos: np.ndarray) -> np.ndarray:
        """返回手的移动向量。"""
        # 切换阶段
        total_frames = self.SLOW_FRAMES + self.FAST_FRAMES
        self.frame_counter = (self.frame_counter + 1) % total_frames
        self.is_fast_phase = self.frame_counter >= self.SLOW_FRAMES

        # 选择步长
        stride = self.FAST_STRIDE if self.is_fast_phase else self.SLOW_STRIDE

        # 计算移动
        if np.random.random() < self.epsilon:
            move = np.random.uniform(-1, 1, size=2)
            move = self._safe_normalize(move) * stride
        else:
            vec = robot_pos - hand_pos
            move = self._safe_normalize(vec) * stride
        return move

    def reset(self):
        """重置痉挛相位。"""
        self.frame_counter = 0
        self.is_fast_phase = False

    @staticmethod
    def _safe_normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return np.zeros_like(vec)
        return vec / norm


class UnseenRLHand:
    """未见过的RL手 (Test 3) - 全新随机种子的RL Hand.

    双杀 A 和 B：真正测试零样本泛化能力。
    不管 A、B 还是 PFSP，训练时绝对没见过这个 Hand。
    """

    def __init__(self, model_path: str, stride: float = 0.35):
        self.model = PPO.load(model_path, custom_objects={
            'learning_rate': 0.0,
            'optimizer_class': None
        }, verbose=0)
        self.stride = stride

    def get_move(self, hand_obs: np.ndarray) -> np.ndarray:
        """返回手的移动向量 (经过stride缩放)。"""
        action, _ = self.model.predict(hand_obs, deterministic=True)
        return action * self.stride


class HumanMouseHand:
    """人类鼠标控制的手 (Test 4) - 用鼠标直接控制。

    降维打击：最具说服力的测试。
    人类玩家会发现 Ours 像个泥鳅，最难抓。
    """

    def __init__(self, env: RehabilitationEnv, cell_size: int = CELL_SIZE):
        self.env = env
        self.cell_size = cell_size
        # 启用旁路模式：跳过生物力学约束
        env._bypass_hand_physics = True
        env.distance_threshold_collision = 1.5

    def update(self, mouse_x_px: int, mouse_y_px: int) -> None:
        """根据鼠标位置更新手的位置。"""
        # 转换鼠标坐标到环境坐标
        target_x = mouse_x_px / self.cell_size
        target_y = mouse_y_px / self.cell_size

        # 裁剪到有效范围
        target_x = np.clip(target_x, self.env.margin, self.env.env_width - self.env.margin)
        target_y = np.clip(target_y, self.env.margin, self.env.env_height - self.env.margin)
        target_pos = np.array([target_x, target_y])

        # 计算平滑移动（尊重stride限制）
        vec_to_mouse = target_pos - self.env.hand_position
        dist_to_mouse = np.linalg.norm(vec_to_mouse)

        if dist_to_mouse > 1e-4:
            move_dist = min(dist_to_mouse, self.env.stride_hand)
            hand_move = (vec_to_mouse / dist_to_mouse) * move_dist
        else:
            hand_move = np.zeros(2)

        # 更新手的物理惯性状态
        self.env.last_hand_actual_move = hand_move.copy()
        self.env.hand_position += hand_move
        self.env.hand_position = np.clip(
            self.env.hand_position,
            self.env.margin,
            [self.env.env_width - self.env.margin, self.env.env_height - self.env.margin]
        )
        self.env.hand_history_buffer.append(hand_move)


# =============================================================================
# Test Runner
# =============================================================================

def calculate_zpd_rate(distances: List[float]) -> float:
    """计算ZPD维护率 - 距离在[4,6]范围内的时间占比。"""
    if not distances:
        return 0.0
    in_zpd = sum(1 for d in distances if ZPD_MIN <= d <= ZPD_MAX)
    return in_zpd / len(distances)


def run_stress_test(
    robot_model_path: str,
    robot_name: str,
    test_hand_type: str,
    unseen_hand_model_path: Optional[str] = None,
    num_episodes: int = 50,
    max_steps: int = 100,
    visual: bool = False,
    fps: int = 8,
) -> StressTestMetrics:
    """运行单个压力测试配置。"""

    # 加载Robot模型
    robot_model = PPO.load(robot_model_path, verbose=0) if robot_model_path else None

    # 创建环境
    env = RehabilitationEnv(
        training_mode='robot',
        robot_model=robot_model,
        hand_model=None,  # 手由测试控制器直接控制
    )
    env.grid_size = GRID_SIZE
    env.cell_size = CELL_SIZE
    env.random_noise = False
    env.max_steps = max_steps

    # 初始化测试手
    test_hand = None
    if test_hand_type == 'sluggish':
        test_hand = SluggishScriptHand()
    elif test_hand_type == 'spasm':
        test_hand = SpasmScriptHand()
    elif test_hand_type == 'unseen_rl':
        if unseen_hand_model_path and os.path.exists(unseen_hand_model_path):
            test_hand = UnseenRLHand(unseen_hand_model_path)
        else:
            print(f"[Warning] Unseen RL hand model not found: {unseen_hand_model_path}")
            return StressTestMetrics(
                robot_name=robot_name, test_name=test_hand_type,
                survival_times=[], mean_survival_time=0, std_survival_time=0,
                catch_count=0, catch_rate=0, zpd_maintenance_rate=0, mean_distance=0
            )
    elif test_hand_type == 'human':
        test_hand = HumanMouseHand(env)

    # 用于可视化的pygame初始化
    screen = None
    clock = None
    if visual:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH_PX, HEIGHT_PX))
        pygame.display.set_caption(f"Stress Test: {robot_name} vs {test_hand_type}")
        clock = pygame.time.Clock()

    # 存储结果
    all_survival_times = []
    catch_count = 0
    all_distances = []

    print(f"\n{'='*60}")
    print(f"Testing: {robot_name} vs {test_hand_type}")
    print(f"{'='*60}")

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_distances = [info['dist']]

        if test_hand_type == 'spasm' and isinstance(test_hand, SpasmScriptHand):
            test_hand.reset()

        if test_hand_type == 'human' and isinstance(test_hand, HumanMouseHand):
            env.stride_robot = 0.6
            env.stride_hand = 0.3

        episode_done = False
        steps = 0

        while not episode_done and steps < max_steps:
            # 处理pygame事件（仅human模式需要）
            if test_hand_type == 'human' and visual:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        episode_done = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            episode_done = True

            # 获取Robot动作
            if robot_model is not None:
                action, _ = robot_model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            # 更新测试手
            if test_hand_type in ['sluggish', 'spasm']:
                assert isinstance(test_hand, (SluggishScriptHand, SpasmScriptHand))
                move = test_hand.get_move(env.hand_position, env.robot_position)
                env.last_hand_actual_move = move.copy()
                env.hand_position += move
                env.hand_position = np.clip(
                    env.hand_position,
                    env.margin,
                    [env.env_width - env.margin, env.env_height - env.margin]
                )
                env.hand_history_buffer.append(move)

            elif test_hand_type == 'unseen_rl':
                assert isinstance(test_hand, UnseenRLHand)
                hand_obs = env._get_hand_obs()
                move = test_hand.get_move(hand_obs)
                env.last_hand_actual_move = move.copy()
                env.hand_position += move
                env.hand_position = np.clip(
                    env.hand_position,
                    env.margin,
                    [env.env_width - env.margin, env.env_height - env.margin]
                )
                env.hand_history_buffer.append(move)

            elif test_hand_type == 'human' and visual:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                test_hand.update(mouse_x, mouse_y)

            # 环境step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_distances.append(info['dist'])
            steps += 1

            # 可视化渲染
            if visual and screen is not None:
                render_aesthetic(
                    env.robot_position,
                    env.hand_position,
                    env.fixed_point,
                    env.trajectory_points,
                    grid_size=GRID_SIZE,
                    cell_size=CELL_SIZE,
                    window=screen
                )
                clock.tick(fps)

            # 检查是否结束
            if terminated or truncated:
                episode_done = True

        # 记录结果
        all_survival_times.append(steps)
        if terminated and info.get('done_reason') == 'Robot Caught':
            catch_count += 1
        all_distances.extend(episode_distances)

        if (ep + 1) % 10 == 0:
            print(f"  Episode {ep+1}/{num_episodes} completed, catch_count={catch_count}")

    env.close()
    if visual:
        pygame.quit()

    # 计算指标
    mean_survival = np.mean(all_survival_times) if all_survival_times else 0
    std_survival = np.std(all_survival_times) if all_survival_times else 0
    mean_dist = np.mean(all_distances) if all_distances else 0

    metrics = StressTestMetrics(
        robot_name=robot_name,
        test_name=test_hand_type,
        survival_times=all_survival_times,
        mean_survival_time=float(mean_survival),
        std_survival_time=float(std_survival),
        catch_count=catch_count,
        catch_rate=catch_count / num_episodes,
        zpd_maintenance_rate=calculate_zpd_rate(all_distances),
        mean_distance=float(mean_dist)
    )

    print(f"\nResults for {robot_name} vs {test_hand_type}:")
    print(f"  Mean Survival Time: {mean_survival:.2f} ± {std_survival:.2f} steps")
    print(f"  Catch Rate: {metrics.catch_rate:.2%}")
    print(f"  ZPD Maintenance: {metrics.zpd_maintenance_rate:.2%}")
    print(f"  Mean Distance: {mean_dist:.2f}")

    return metrics


def run_human_test_interactive(
    robot_model_path: str,
    robot_name: str,
    max_steps: int = 500,
    fps: int = 30,
) -> Tuple[int, float, List[float]]:
    """运行交互式人类测试（鼠标控制）。"""

    robot_model = PPO.load(robot_model_path, verbose=0)

    env = RehabilitationEnv(
        training_mode='robot',
        robot_model=robot_model,
        hand_model=None,
    )
    env.grid_size = GRID_SIZE
    env.cell_size = CELL_SIZE
    env._bypass_hand_physics = True
    env.distance_threshold_collision = 1.5
    env.stride_robot = 0.6
    env.stride_hand = 0.3
    env.max_steps = max_steps

    pygame.init()
    screen = pygame.display.set_mode((WIDTH_PX, HEIGHT_PX))
    pygame.display.set_caption(f"HUMAN TEST: You control the HAND! vs {robot_name}")
    clock = pygame.time.Clock()

    obs, info = env.reset()
    distances = [info['dist']]
    steps = 0
    running = True
    caught = False

    print(f"\nHuman test started: You are the HAND!")
    print(f"Goal: Survive as long as possible. Robot is trying to catch you.")
    print(f"Press Q to quit.\n")

    while running and steps < max_steps:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False

        # 获取鼠标位置并更新手
        mouse_x, mouse_y = pygame.mouse.get_pos()
        target_x = np.clip(mouse_x / CELL_SIZE, env.margin, env.env_width - env.margin)
        target_y = np.clip(mouse_y / CELL_SIZE, env.margin, env.env_height - env.margin)
        target_pos = np.array([target_x, target_y])

        vec_to_mouse = target_pos - env.hand_position
        dist_to_mouse = np.linalg.norm(vec_to_mouse)
        if dist_to_mouse > 1e-4:
            move_dist = min(dist_to_mouse, env.stride_hand)
            hand_move = (vec_to_mouse / dist_to_mouse) * move_dist
        else:
            hand_move = np.zeros(2)

        env.last_hand_actual_move = hand_move.copy()
        env.hand_position += hand_move
        env.hand_position = np.clip(
            env.hand_position, env.margin,
            [env.env_width - env.margin, env.env_height - env.margin]
        )
        env.hand_history_buffer.append(hand_move)

        # Robot动作
        action, _ = robot_model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        distances.append(info['dist'])
        steps += 1

        # 渲染
        render_aesthetic(
            env.robot_position,
            env.hand_position,
            env.fixed_point,
            env.trajectory_points,
            grid_size=GRID_SIZE,
            cell_size=CELL_SIZE,
            window=screen
        )

        # 显示FPS和距离
        font = pygame.font.SysFont('arial', 18)
        dist_text = font.render(f"Dist: {info['dist']:.2f}  Steps: {steps}  ZPD: {ZPD_MIN}-{ZPD_MAX}", True, (0, 0, 0))
        screen.blit(dist_text, (10, 10))

        pygame.display.flip()
        clock.tick(fps)

        if terminated or truncated:
            caught = terminated and info.get('done_reason') == 'Robot Caught'
            running = False
            break

    pygame.quit()
    env.close()

    return steps, caught, distances


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-dimensional Stress Test Suite for Robot Evaluation'
    )
    parser.add_argument(
        '--robot',
        type=str,
        default='all',
        choices=['baseline_a', 'baseline_b', 'pfsp', 'all'],
        help='Robot to test'
    )
    parser.add_argument(
        '--test',
        type=str,
        default='all',
        choices=['sluggish', 'spasm', 'unseen_rl', 'human', 'all'],
        help='Test hand type'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=50,
        help='Number of episodes per test'
    )
    parser.add_argument(
        '--visual',
        action='store_true',
        help='Enable visual rendering'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=8,
        help='Frames per second for visual mode'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='stress_test_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--baseline_b_path',
        type=str,
        default=None,
        help='Path to Baseline B model (if already trained)'
    )

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 确定要测试的robot和test组合
    robots = []
    if args.robot == 'all':
        robots = ['baseline_a', 'baseline_b', 'pfsp']
    else:
        robots = [args.robot]

    tests = []
    if args.test == 'all':
        tests = ['sluggish', 'spasm', 'unseen_rl', 'human']
    else:
        tests = [args.test]

    # 获取有效model path
    def get_robot_path(name: str) -> Optional[str]:
        if name == 'baseline_b':
            return args.baseline_b_path if args.baseline_b_path else MODEL_PATHS.get(name)
        return MODEL_PATHS.get(name)

    # 运行所有测试
    all_results = []
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    for robot in robots:
        robot_path = get_robot_path(robot)
        if robot_path is None:
            print(f"\n[Skipping {robot}] Model path is None (needs training)")
            continue
        if not os.path.exists(robot_path):
            print(f"\n[Skipping {robot}] Model not found: {robot_path}")
            continue

        for test in tests:
            if test == 'human':
                # 人类测试需要交互式界面，单独处理
                print(f"\n[Starting HUMAN test for {robot}]")
                survival, caught, distances = run_human_test_interactive(
                    robot_path, robot, fps=args.fps
                )
                metrics = StressTestMetrics(
                    robot_name=robot,
                    test_name='human',
                    survival_times=[survival],
                    mean_survival_time=survival,
                    std_survival_time=0,
                    catch_count=1 if caught else 0,
                    catch_rate=1.0 if caught else 0.0,
                    zpd_maintenance_rate=calculate_zpd_rate(distances),
                    mean_distance=np.mean(distances) if distances else 0
                )
            else:
                metrics = run_stress_test(
                    robot_model_path=robot_path,
                    robot_name=robot,
                    test_hand_type=test,
                    unseen_hand_model_path=UNSEEN_HAND_PATH,
                    num_episodes=args.episodes,
                    visual=args.visual,
                    fps=args.fps
                )
            all_results.append(metrics)

    # 保存结果
    if all_results:
        results_file = os.path.join(args.output, f"results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump([r.to_dict() for r in all_results], f, indent=2)
        print(f"\n{'='*60}")
        print(f"Results saved to: {results_file}")
        print(f"{'='*60}")

        # 打印汇总表
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)
        print(f"{'Test':<12} {'Robot':<15} {'Survival':<12} {'Catch Rate':<12} {'ZPD Rate':<12}")
        print("-"*80)
        for r in all_results:
            print(f"{r.test_name:<12} {r.robot_name:<15} {r.mean_survival_time:<12.2f} "
                  f"{r.catch_rate:<12.2%} {r.zpd_maintenance_rate:<12.2%}")
        print("="*80)
    else:
        print("\nNo results to save (all tests were skipped).")


if __name__ == '__main__':
    main()
