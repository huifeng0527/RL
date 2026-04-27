"""Dual agent iterative training.

Alternating training between robot agent and hand agent.
Robot learns to catch hand, hand learns to avoid robot.
"""

import sys
import os

# Add project root to path for direct script execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import argparse
import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from src.custom_env import RehabilitationEnv
from src.utils.callbacks import DebugCallback
from src.utils.ablation_callbacks import PPOAuxTrainingCallback
from src.utils.feature_extractors import MLPOnlyExtractor, LSTMExtractor, AuxLSTMExtractor, GatedExtractor, AuxGatedExtractor

EXTRACTOR_MAP = {
    "mlp": MLPOnlyExtractor,
    "lstm": LSTMExtractor,
    "aux_lstm": AuxLSTMExtractor,
    "gate": GatedExtractor,
    "aux_gate": AuxGatedExtractor,
}

# Check if extractor has auxiliary task
AUX_EXTRACTORS = {AuxLSTMExtractor, AuxGatedExtractor}


def _copy_models_from_resume(resume_from, base_save_path, start_from_iteration):
    """复制 resume_from 中的模型到新目录，以便下次 resume。

    这样即使训练中断，下次 resume 时也不会丢失之前的模型。
    """
    import shutil

    copied_count = 0
    for i in range(1, start_from_iteration):
        src_iter = os.path.join(resume_from, f"iteration_{i}")
        dst_iter = os.path.join(base_save_path, f"iteration_{i}")

        # 复制 Robot 模型
        src_robot = os.path.join(src_iter, "robot", "robot", "best_model.zip")
        dst_robot = os.path.join(dst_iter, "robot", "robot")
        if os.path.exists(src_robot):
            os.makedirs(dst_robot, exist_ok=True)
            shutil.copy2(src_robot, os.path.join(dst_robot, "best_model.zip"))
            # 也复制 final_model 如果存在
            src_final = os.path.join(src_iter, "robot", "robot", "final_model.zip")
            if os.path.exists(src_final):
                shutil.copy2(src_final, os.path.join(dst_robot, "final_model.zip"))
            print(f"    [Copy] Robot Iter {i}")
            copied_count += 1

        # 复制 Hand 模型
        src_hand = os.path.join(src_iter, "hand", "hand", "best_model.zip")
        dst_hand = os.path.join(dst_iter, "hand", "hand")
        if os.path.exists(src_hand):
            os.makedirs(dst_hand, exist_ok=True)
            shutil.copy2(src_hand, os.path.join(dst_hand, "best_model.zip"))
            print(f"    [Copy] Hand Iter {i}")
            copied_count += 1

    return copied_count


def create_env(training_mode, robot_model=None, hand_model_paths=None):
    """Create a single environment instance."""
    def _make():
        env = RehabilitationEnv(
            training_mode=training_mode,
            robot_model=robot_model,
            hand_model_paths=hand_model_paths
        )
        return Monitor(env)
    return _make


def create_vec_env(training_mode, n_envs=4, robot_model=None, hand_model_paths=None):
    """Create vectorized environment."""
    env_fns = [create_env(training_mode, robot_model, hand_model_paths) for _ in range(n_envs)]
    return DummyVecEnv(env_fns)


def train_robot(hand_model_paths, total_steps, save_path, n_envs=4, extractor_class=None,
                previous_robot_path=None, continue_from_previous=False, start_iteration=1,
                skip_if_exists=False, resume_from_path=None):
    """Train robot agent to catch hand.

    Args:
        continue_from_previous: If True, load and continue training from previous_robot_path
        start_iteration: The iteration number to start from (affects learning rate schedule)
        skip_if_exists: If True, skip training if best_model.zip already exists
        resume_from_path: Path to existing model to use (for resume scenario)
    """
    # 检查是否跳过训练
    robot_model_path = os.path.join(save_path, "robot", "best_model.zip")
    if skip_if_exists:
        if os.path.exists(robot_model_path):
            print(f"\n{'='*40}")
            print("Skipping Robot Agent (model already exists at save_path)")
            print(f"{'='*40}")
            return robot_model_path
        elif resume_from_path and os.path.exists(resume_from_path):
            print(f"\n{'='*40}")
            print("Skipping Robot Agent (using model from resume_from)")
            print(f"{'='*40}")
            return resume_from_path

    # 确定使用哪个已有模型继续训练
    if resume_from_path and os.path.exists(resume_from_path):
        existing_model = resume_from_path
    elif continue_from_previous and previous_robot_path and os.path.exists(previous_robot_path):
        existing_model = previous_robot_path
    else:
        existing_model = None

    print(f"\n{'='*40}")
    print("Training Robot Agent")
    print(f"{'='*40}")

    vec_env = create_vec_env('robot', n_envs=n_envs, hand_model_paths=hand_model_paths)

    policy_kwargs = dict(
        net_arch=[256, 256, 256, 64],
        share_features_extractor=True
    )
    if extractor_class is not None:
        policy_kwargs["features_extractor_class"] = extractor_class

    # 加载并继续训练
    if existing_model:
        print(f"[*] 继续训练: 加载 Robot 模型 -> {existing_model}")
        # 使用较低学习率继续训练
        model = PPO.load(
            existing_model,
            env=vec_env,
            custom_objects={
                'learning_rate': 1e-4,  # 降低学习率继续训练
                'ent_coef': 0.008,
                'n_epochs': 4,
                'max_grad_norm': 0.5,
                'batch_size': 1024,
                'n_steps': 4096,
            },
            verbose=0
        )
        # 重新设置学习率
        model.learning_rate = 1e-4
        logger = configure(os.path.join(save_path, "tensorboard"), ["tensorboard"])
        model.set_logger(logger)
    else:
        print("[*] 初始训练: 创建全新的 Robot 模型")
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            learning_rate=3e-4,
            ent_coef=0.01,
            n_epochs=4,
            max_grad_norm=0.5,
            batch_size=1024,
            n_steps=4096,
            policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(save_path, "tensorboard")
        )

    eval_env = create_vec_env('robot', n_envs=1, hand_model_paths=hand_model_paths)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, "robot"),
        eval_freq=10000,
        deterministic=True,
        n_eval_episodes=100
    )
    # debug_callback = DebugCallback(env=eval_env, log_freq=10000)

    # 添加 Aux callback 如果使用 aux extractor
    use_aux = extractor_class in AUX_EXTRACTORS
    if use_aux:
        print("[*] 使用 Aux 训练回调")
        aux_callback = PPOAuxTrainingCallback(batch_size=512)
        callbacks = CallbackList([eval_callback, aux_callback])
    else:
        callbacks = CallbackList([eval_callback])

    model.learn(
        total_timesteps=total_steps,
        callback=callbacks
    )

    model.save(robot_model_path)
    print(f"Robot model saved to {robot_model_path}")

    vec_env.close()
    eval_env.close()

    return robot_model_path


def train_hand(robot_model_path, total_steps, save_path, n_envs=4, extractor_class=None,
               skip_if_exists=False, resume_from_path=None, warm_start_path=None):
    """Train hand agent to avoid robot.

    Args:
        skip_if_exists: If True, skip training if best_model.zip already exists
        resume_from_path: Path to existing model to skip training (use directly)
        warm_start_path: Path to previous Hand model to warm-start from (continue training)
    """
    # 检查是否跳过训练
    hand_model_path = os.path.join(save_path, "hand", "best_model.zip")
    if skip_if_exists:
        if os.path.exists(hand_model_path):
            print(f"\n{'='*40}")
            print("Skipping Hand Agent (model already exists at save_path)")
            print(f"{'='*40}")
            return hand_model_path
        elif resume_from_path and os.path.exists(resume_from_path):
            print(f"\n{'='*40}")
            print("Skipping Hand Agent (using model from resume_from)")
            print(f"{'='*40}")
            return resume_from_path

    print(f"\n{'='*40}")
    print("Training Hand Agent")
    print(f"{'='*40}")

    # Load robot model as opponent
    robot_model = PPO.load(
        robot_model_path,
        custom_objects={'learning_rate': 0.0, 'optimizer_class': None},
        verbose=0
    )

    vec_env = create_vec_env('hand', n_envs=n_envs, robot_model=robot_model)

    policy_kwargs = dict(
        net_arch=[256, 256, 64],
        share_features_extractor=True
    )
    if extractor_class is not None:
        policy_kwargs["features_extractor_class"] = extractor_class

    # 热启动：从上一代 Hand 模型继续训练（保留通用能力）
    if warm_start_path and os.path.exists(warm_start_path):
        print(f"[*] 热启动: 从上一代 Hand 模型继续训练 -> {warm_start_path}")
        model = PPO.load(
            warm_start_path,
            env=vec_env,
            custom_objects={
                'learning_rate': 1e-4,  # 降低学习率微调
                'ent_coef': 0.008,
                'n_epochs': 4,
                'max_grad_norm': 0.5,
                'batch_size': 512,
                'n_steps': 2048,
            },
            verbose=0
        )
        model.learning_rate = 1e-4
        logger = configure(os.path.join(save_path, "tensorboard"), ["tensorboard"])
        model.set_logger(logger)
    else:
        print("[*] 基因变异: 创建全新的 Hand 对手模型 (从头开始)")
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            learning_rate=3e-4,
            ent_coef=0.01,
            n_epochs=4,
            max_grad_norm=0.5,
            batch_size=512,
            n_steps=2048,
            policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(save_path, "tensorboard")
        )

    eval_env = create_vec_env('hand', n_envs=1, robot_model=robot_model)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, "hand"),
        eval_freq=10000,
        deterministic=True,
        n_eval_episodes=100
    )
    debug_callback = DebugCallback(env=eval_env, log_freq=10000)

    # 添加 Aux callback 如果使用 aux extractor
    use_aux = extractor_class in AUX_EXTRACTORS
    if use_aux:
        print("[*] 使用 Aux 训练回调")
        aux_callback = PPOAuxTrainingCallback(batch_size=512)
        callbacks = CallbackList([eval_callback, aux_callback, debug_callback])
    else:
        callbacks = CallbackList([eval_callback, debug_callback])

    model.learn(
        total_timesteps=total_steps,
        callback=callbacks
    )

    hand_path = os.path.join(save_path, "hand", "best_model.zip")
    model.save(hand_path)
    print(f"Hand model saved to {hand_path}")

    vec_env.close()
    eval_env.close()

    return hand_path


def run_iterative_training(
    n_iterations=5,
    robot_steps=1_000_000,
    hand_steps=2_000_000,
    n_envs=4,
    base_save_path=None,
    extractor_name="mlp",
    resume_from=None,
    start_from_iteration=1
):
    """Run iterative dual agent training.

    Args:
        resume_from: Path to previous training run to continue from (e.g., "logs/dual_iterative_0419_1041")
        start_from_iteration: Start training from this iteration number (e.g., 2 to continue from iteration 1's models)
    """
    if base_save_path is None:
        now = datetime.datetime.now().strftime("%m%d_%H%M")
        base_save_path = f"logs/dual_iterative_{now}"

    os.makedirs(base_save_path, exist_ok=True)

    # Get extractor class
    extractor_class = EXTRACTOR_MAP.get(extractor_name.lower())
    if extractor_class is None:
        print(f"Warning: Unknown extractor '{extractor_name}', using default (no custom extractor)")
        extractor_class = None
    else:
        print(f"Using feature extractor: {extractor_class.__name__}")

    # 加载历史手模型路径
    historical_hand_paths = []

    # 如果从之前的训练继续
    current_robot_path = None
    if resume_from and os.path.exists(resume_from):
        print(f"\n[*] 从之前的训练继续: {resume_from}")
        print(f"[*] 从第 {start_from_iteration} 轮开始继续\n")

        # 加载之前所有轮次的手模型到对手池
        for i in range(1, start_from_iteration):
            prev_hand_path = os.path.join(resume_from, f"iteration_{i}", "hand", "hand", "best_model.zip")
            if os.path.exists(prev_hand_path):
                historical_hand_paths.append(prev_hand_path)
                print(f"    已加载 Hand Iter {i} -> 对手池")

        # 加载上一轮的 Robot 模型作为起点
        if start_from_iteration > 1:
            
            current_robot_path = os.path.join(resume_from, f"iteration_{start_from_iteration - 1}", "robot", "robot", "best_model.zip")
            if not os.path.exists(current_robot_path):
                # 尝试 final_model
                current_robot_path = os.path.join(resume_from, f"iteration_{start_from_iteration - 1}", "robot", "robot", "final_model.zip")
            if os.path.exists(current_robot_path):
                print(f"    已加载 Robot Iter {start_from_iteration - 1} -> 继续训练")

        # 复制之前所有迭代的模型到新目录（以便下次 resume）
        print("\n[*] 复制历史模型到新目录...")
        copied = _copy_models_from_resume(resume_from, base_save_path, start_from_iteration)
        if copied > 0:
            print(f"    已复制 {copied} 个模型文件\n")

        # 更新 historical_hand_paths 指向新复制的路径
        historical_hand_paths = []
        for i in range(1, start_from_iteration):
            new_hand_path = os.path.join(base_save_path, f"iteration_{i}", "hand", "hand", "best_model.zip")
            if os.path.exists(new_hand_path):
                historical_hand_paths.append(new_hand_path)

    print(f"➤ 初始对手池大小: {len(historical_hand_paths) + 1} (包含脚本手)")

    # 从 start_from_iteration 开始训练
    for iteration in range(start_from_iteration, n_iterations + 1):
        print(f"\n{'#'*60}")
        print(f"# Iteration {iteration}/{n_iterations}")
        print(f"{'#'*60}")

        iteration_path = os.path.join(base_save_path, f"iteration_{iteration}")
        os.makedirs(iteration_path, exist_ok=True)

        # 检查当前 iteration 是否已有模型
        resume_robot_path = None
        # 只有当 iteration < start_from_iteration 时才加载已有模型（之前的轮次）
        # start_from_iteration 必须重新训练
        if resume_from and iteration < start_from_iteration:
            prev_robot = os.path.join(resume_from, f"iteration_{iteration}", "robot", "robot", "best_model.zip")
            if os.path.exists(prev_robot):
                resume_robot_path = prev_robot
                print(f"    发现已有 Robot Iter {iteration}")

        # 判断是否从上一轮继续训练
        continue_from_previous = (iteration > start_from_iteration) or (resume_from is not None and iteration == start_from_iteration)
        # 只有当没有从 resume_from 找到模型时才用 previous_robot_path
        if resume_robot_path is None and continue_from_previous and current_robot_path and os.path.exists(current_robot_path):
            resume_robot_path = current_robot_path

        print(f"➤ 当前对手池大小: {len(historical_hand_paths) + 1}")

        # 1. 训练 Robot
        current_robot_path = train_robot(
            hand_model_paths=historical_hand_paths,
            total_steps=robot_steps,
            save_path=os.path.join(iteration_path, "robot"),
            n_envs=n_envs,
            extractor_class=extractor_class,
            previous_robot_path=current_robot_path,
            continue_from_previous=continue_from_previous and (current_robot_path is not None),
            start_iteration=iteration,
            skip_if_exists=False,  # start_from 指定的轮次必须训练
            resume_from_path=resume_robot_path  # 之前的轮次从这里加载
        )

        # 检查当前 iteration 是否已有 Hand 模型
        resume_hand_path = None
        # 只有当 iteration < start_from_iteration 时才加载已有模型（之前的轮次）
        if resume_from and iteration < start_from_iteration:
            prev_hand = os.path.join(resume_from, f"iteration_{iteration}", "hand", "hand", "best_model.zip")
            if os.path.exists(prev_hand):
                resume_hand_path = prev_hand
                print(f"    发现已有 Hand Iter {iteration}")

        # 2. 训练 Hand
        # 热启动：从上一轮 Hand 模型继续训练
        prev_hand_warm_start = None
        if iteration > 1:
            prev_hand_path = os.path.join(base_save_path, f"iteration_{iteration - 1}", "hand", "hand", "best_model.zip")
            if os.path.exists(prev_hand_path):
                prev_hand_warm_start = prev_hand_path

        new_hand_path = train_hand(
            robot_model_path=current_robot_path,
            total_steps=hand_steps,
            save_path=os.path.join(iteration_path, "hand"),
            n_envs=n_envs,
            extractor_class=extractor_class,
            skip_if_exists=False,  # start_from 指定的轮次必须训练
            resume_from_path=resume_hand_path,  # 之前的轮次（跳过训练时使用）
            warm_start_path=prev_hand_warm_start  # 上一轮 Hand 模型（热启动）
        )

        # 如果跳过了训练，使用 resume 的路径
        if new_hand_path is None or not os.path.exists(new_hand_path):
            if resume_hand_path:
                new_hand_path = resume_hand_path
                print(f"    复用已有 Hand 模型: {new_hand_path}")

        # 3. 将新 Hand 加入对手池
        if new_hand_path not in historical_hand_paths:
            historical_hand_paths.append(new_hand_path)
            print(f"[*] 已将 Hand Iter {iteration} 添加入对手池")

    print(f"\n{'='*60}")
    print("Iterative League Training Complete!")
    print(f"{'='*60}")
    print(f"Final Robot: {current_robot_path}")
    print(f"Total Opponents: {len(historical_hand_paths)}")

    return current_robot_path, historical_hand_paths[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual agent iterative league training')
    parser.add_argument('--iterations', type=int, default=10, help='Number of training iterations')
    parser.add_argument('--robot_steps', type=int, default=1_000_000, help='Steps per iteration for robot')
    parser.add_argument('--hand_steps', type=int, default=2_000_000, help='Steps per iteration for hand')
    parser.add_argument('--n_envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--save_path', type=str, default=None, help='Save path')
    parser.add_argument('--extractor', type=str, default='aux_lstm',
                        choices=['mlp', 'lstm', 'aux_lstm', 'gate', 'aux_gate'],
                        help='Feature extractor to use')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to previous training run to continue from (e.g., logs/dual_iterative_0419_1041)')
    parser.add_argument('--start_from', type=int, default=1,
                        help='Start from this iteration number (e.g., 2 to continue from iteration 1)')

    args = parser.parse_args()

    run_iterative_training(
        n_iterations=args.iterations,
        robot_steps=args.robot_steps,
        hand_steps=args.hand_steps,
        n_envs=args.n_envs,
        base_save_path=args.save_path,
        extractor_name=args.extractor,
        resume_from=args.resume_from,
        start_from_iteration=args.start_from
    )
