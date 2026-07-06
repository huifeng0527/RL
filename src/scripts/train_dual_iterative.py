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
import json
import math
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from src.custom_env import RehabilitationEnv
from src.observation_schema import HISTORY_CHANNELS, INTERACTION_HISTORY_CHANNELS
from src.scripts.cross_eval import evaluate_pair
from src.utils.callbacks import DebugCallback, StatusWriter, TrainingStatusCallback
from src.utils.ablation_callbacks import PPOAuxTrainingCallback, PPOFutureAuxTrainingCallback
from src.utils.feature_extractors import (
    MLPOnlyExtractor,
    StrategyGRUAuxExtractor,
)

EXTRACTOR_MAP = {
    "mlp": MLPOnlyExtractor,
    "gru": StrategyGRUAuxExtractor,
}

GRU_EXTRACTORS = {StrategyGRUAuxExtractor}


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


def create_env(training_mode, robot_model=None, hand_model_paths=None, include_opponent_id=False,
               scripted_hand_sample_prob=None, robot_opponent_id_dim=0, history_length=16,
               history_mode="motion"):
    """Create a single environment instance."""
    def _make():
        env = RehabilitationEnv(
            training_mode=training_mode,
            robot_model=robot_model,
            hand_model_paths=hand_model_paths,
            include_opponent_id=include_opponent_id if training_mode == 'robot' else False,
            robot_opponent_id_dim=robot_opponent_id_dim if training_mode == 'hand' else 0,
            history_length=history_length,
            history_mode=history_mode if training_mode == 'robot' else "motion",
        )
        if scripted_hand_sample_prob is not None:
            env.scripted_hand_sample_prob = scripted_hand_sample_prob
        return Monitor(env)
    return _make


def create_vec_env(training_mode, n_envs=4, robot_model=None, hand_model_paths=None,
                   include_opponent_id=False, scripted_hand_sample_prob=None,
                   robot_opponent_id_dim=0, history_length=16, history_mode="motion"):
    """Create vectorized environment."""
    env_fns = [
        create_env(
            training_mode,
            robot_model,
            hand_model_paths,
            include_opponent_id,
            scripted_hand_sample_prob,
            robot_opponent_id_dim,
            history_length,
            history_mode,
        )
        for _ in range(n_envs)
    ]
    return DummyVecEnv(env_fns)


def train_robot(hand_model_paths, total_steps, save_path, n_envs=4, extractor_class=None,
                previous_robot_path=None, continue_from_previous=False, start_iteration=1,
                skip_if_exists=False, resume_from_path=None, status_run_dir=None,
                status_log_freq=10000, include_opponent_id=False,
                scripted_hand_sample_prob=None, history_length=16,
                history_mode="motion", future_horizon=8, aux_mode="none",
                strategy_aux_weights=None):
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

    vec_env = create_vec_env(
        'robot',
        n_envs=n_envs,
        hand_model_paths=hand_model_paths,
        include_opponent_id=include_opponent_id,
        scripted_hand_sample_prob=scripted_hand_sample_prob,
        history_length=history_length,
        history_mode=history_mode,
    )
    obs_dim = vec_env.observation_space.shape[0]
    history_channels = INTERACTION_HISTORY_CHANNELS if history_mode == "interaction" else HISTORY_CHANNELS
    base_obs_dim = 12 + history_length * history_channels
    opponent_id_dim = max(0, obs_dim - base_obs_dim)
    print(
        f"[*] Robot obs_dim={obs_dim}, history_length={history_length}, "
        f"include_opponent_id={include_opponent_id}, opponent_id_dim={opponent_id_dim}"
    )

    policy_kwargs = dict(
        net_arch=[256, 256, 256, 64],
        share_features_extractor=True
    )
    if extractor_class is not None:
        policy_kwargs["features_extractor_class"] = extractor_class
        if extractor_class in GRU_EXTRACTORS:
            policy_kwargs["features_extractor_kwargs"] = dict(
                future_horizon=future_horizon,
                history_channels=history_channels,
                history_length=history_length,
            )

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
        tensorboard_dir = os.path.join(save_path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        logger = configure(tensorboard_dir, ["tensorboard"])
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

    eval_env = create_vec_env(
        'robot',
        n_envs=1,
        hand_model_paths=hand_model_paths,
        include_opponent_id=include_opponent_id,
        scripted_hand_sample_prob=scripted_hand_sample_prob,
        history_length=history_length,
        history_mode=history_mode,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, "robot"),
        eval_freq=10000,
        deterministic=True,
        n_eval_episodes=100
    )
    # debug_callback = DebugCallback(env=eval_env, log_freq=10000)

    # 添加 callbacks
    callback_items = [eval_callback]
    if status_run_dir is not None:
        callback_items.append(TrainingStatusCallback(
            run_dir=status_run_dir,
            phase="robot",
            iteration=start_iteration,
            total_timesteps=total_steps,
            log_freq=status_log_freq,
        ))

    aux_mode = str(aux_mode).lower()
    if aux_mode == "single":
        print("[*] Using single-step auxiliary callback")
        callback_items.append(PPOAuxTrainingCallback(
            batch_size=512,
            history_length=history_length,
            history_channels=history_channels,
        ))
    elif aux_mode in {"multi_risk", "contrastive"}:
        print(f"[*] Using {aux_mode} strategy auxiliary callback")
        weights = strategy_aux_weights or {}
        contrastive_weight = weights.get("contrastive", 0.05) if aux_mode == "contrastive" else 0.0
        callback_items.append(PPOFutureAuxTrainingCallback(
            history_length=history_length,
            history_channels=history_channels,
            future_horizon=future_horizon,
            batch_size=512,
            traj_weight=weights.get("traj", 1.0),
            risk_weight=weights.get("risk", 0.2),
            contrastive_weight=contrastive_weight,
            contrastive_temperature=weights.get("temperature", 0.1),
        ))

    callbacks = CallbackList(callback_items)

    model.learn(
        total_timesteps=total_steps,
        callback=callbacks
    )

    final_robot_model_path = os.path.join(save_path, "robot", "final_model.zip")
    model.save(final_robot_model_path)
    print(f"Robot final model saved to {final_robot_model_path}")
    if os.path.exists(robot_model_path):
        print(f"Robot best model kept at {robot_model_path}")
    else:
        robot_model_path = final_robot_model_path

    vec_env.close()
    eval_env.close()

    return robot_model_path


def train_hand(robot_model_path, total_steps, save_path, n_envs=4, extractor_class=None,
               skip_if_exists=False, resume_from_path=None, warm_start_path=None,
               status_run_dir=None, status_log_freq=10000, iteration=1,
               robot_opponent_id_dim=0):
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

    vec_env = create_vec_env(
        'hand',
        n_envs=n_envs,
        robot_model=robot_model,
        robot_opponent_id_dim=robot_opponent_id_dim,
    )

    policy_kwargs = dict(
        net_arch=[256, 256, 64],
        share_features_extractor=True,
        features_extractor_class=StrategyGRUAuxExtractor,
        features_extractor_kwargs=dict(
            future_horizon=1,
            history_channels=HISTORY_CHANNELS,
            history_length=16,
        ),
    )
    print("[*] Hand extractor fixed to GRU")

    # 热启动：从上一代 Hand 模型继续训练（保留通用能力）
    if warm_start_path and os.path.exists(warm_start_path):
        print(f"[*] 热启动: 从上一代 Hand 模型继续训练 -> {warm_start_path}")
        model = PPO.load(
            warm_start_path,
            env=vec_env,
            custom_objects={
                'learning_rate': 1e-4,  # 降低学习率微调
                'ent_coef': 0.03,
                'n_epochs': 4,
                'max_grad_norm': 0.5,
                'batch_size': 512,
                'n_steps': 2048,
            },
            verbose=0
        )
        model.learning_rate = 1e-4
        model.ent_coef = 0.03
        tensorboard_dir = os.path.join(save_path, "tensorboard")
        os.makedirs(tensorboard_dir, exist_ok=True)
        logger = configure(tensorboard_dir, ["tensorboard"])
        model.set_logger(logger)
    else:
        print("[*] 基因变异: 创建全新的 Hand 对手模型 (从头开始)")
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            learning_rate=3e-4,
            ent_coef=0.03,
            n_epochs=4,
            max_grad_norm=0.5,
            batch_size=512,
            n_steps=2048,
            policy_kwargs=policy_kwargs,
            tensorboard_log=os.path.join(save_path, "tensorboard")
        )

    eval_env = create_vec_env(
        'hand',
        n_envs=1,
        robot_model=robot_model,
        robot_opponent_id_dim=robot_opponent_id_dim,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_path, "hand"),
        eval_freq=10000,
        deterministic=True,
        n_eval_episodes=100
    )
    debug_callback = DebugCallback(env=eval_env, log_freq=10000)

    # 添加 callbacks
    callback_items = [eval_callback, debug_callback]
    if status_run_dir is not None:
        callback_items.append(TrainingStatusCallback(
            run_dir=status_run_dir,
            phase="hand",
            iteration=iteration,
            total_timesteps=total_steps,
            log_freq=status_log_freq,
        ))

    callbacks = CallbackList(callback_items)

    model.learn(
        total_timesteps=total_steps,
        callback=callbacks
    )

    hand_path = os.path.join(save_path, "hand", "best_model.zip")
    final_hand_path = os.path.join(save_path, "hand", "final_model.zip")
    model.save(final_hand_path)
    print(f"Hand final model saved to {final_hand_path}")
    if os.path.exists(hand_path):
        print(f"Hand best model kept at {hand_path}")
    else:
        hand_path = final_hand_path

    vec_env.close()
    eval_env.close()

    return hand_path

def evaluate_convergence(robot_path, hand_paths, episodes=100, max_steps=40, zpd_min=4.0, zpd_max=6.0,
                         pool_score_lambda=0.25):
    pair_metrics = []
    for idx, hand_path in enumerate(hand_paths, start=1):
        metrics = evaluate_pair(
            robot_path,
            hand_path,
            num_episodes=episodes,
            max_steps=max_steps,
            z_min=zpd_min,
            z_max=zpd_max,
        )
        metrics.update({"hand_index": idx, "hand_path": hand_path})
        pair_metrics.append(metrics)

    if not pair_metrics:
        return None

    tis_means = np.array([m["tis_mean"] for m in pair_metrics], dtype=float)
    tis_ses = np.array([
        m["tis_std"] / math.sqrt(max(m.get("num_episodes", episodes), 1))
        for m in pair_metrics
    ], dtype=float)
    zpd_means = np.array([m["zpd_coverage_mean"] for m in pair_metrics], dtype=float)
    lengths = np.array([m["episode_length_mean"] for m in pair_metrics], dtype=float)

    aggregate_se = float(math.sqrt(float(np.sum(tis_ses ** 2))) / len(pair_metrics))
    mean_tis = float(np.mean(tis_means))
    std_tis = float(np.std(tis_means))
    pool_score = float(mean_tis - pool_score_lambda * std_tis)

    return {
        "num_opponents": len(pair_metrics),
        "episodes_per_opponent": episodes,
        "mean_tis": mean_tis,
        "std_tis_across_hands": std_tis,
        "worst_case_tis": float(np.min(tis_means)),
        "aggregate_se": aggregate_se,
        "mean_zpd_coverage": float(np.mean(zpd_means)),
        "mean_episode_length": float(np.mean(lengths)),
        "pool_score": pool_score,
        "pool_score_lambda": pool_score_lambda,
        "pair_metrics": pair_metrics,
    }


def convergence_decision(current, previous, stagnant_count, patience=2, z_value=1.96):
    if previous is None or current is None:
        return {
            "stop": False,
            "stagnant_count": 0,
            "reason": "insufficient_history",
        }

    delta = current["pool_score"] - previous["pool_score"]
    delta_se = math.sqrt(current["aggregate_se"] ** 2 + previous["aggregate_se"] ** 2)
    noise_band = z_value * delta_se
    stagnant = delta <= noise_band
    next_count = stagnant_count + 1 if stagnant else 0
    return {
        "stop": next_count >= patience,
        "stagnant_count": next_count,
        "reason": "improvement_within_statistical_noise" if stagnant else "improving_beyond_noise",
        "delta": float(delta),
        "delta_se": float(delta_se),
        "noise_band": float(noise_band),
        "z_value": z_value,
        "patience": patience,
    }


def append_json(path, payload):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_iterative_training(
    n_iterations=5,
    robot_steps=2_000_000,
    first_robot_steps=3_000_000,
    hand_steps=1_000_000,
    n_envs=4,
    base_save_path=None,
    extractor_name="mlp",
    resume_from=None,
    start_from_iteration=1,
    enable_convergence_stop=False,
    convergence_episodes=100,
    convergence_patience=2,
    convergence_z=1.96,
    pool_score_lambda=0.25,
    status_log_freq=10000,
    include_opponent_id=False,
    scripted_hand_sample_prob=None,
    aux_mode="none",
    history_length=16,
    history_mode="motion",
    future_horizon=8,
    strategy_traj_weight=0.1,
    strategy_risk_weight=0.02,
    strategy_contrastive_weight=0.05,
    strategy_contrastive_temperature=0.1,
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
    extractor_name = extractor_name.lower()
    aux_mode = aux_mode.lower()
    if extractor_name == "gru" and history_mode == "motion":
        print("[*] GRU league experiments use interaction history; setting history_mode=interaction.")
        history_mode = "interaction"
    if extractor_name == "mlp" and aux_mode != "none":
        raise ValueError("MLP baseline only supports --aux none")

    status_writer = StatusWriter(base_save_path)
    convergence_history_path = os.path.join(base_save_path, "convergence_history.jsonl")
    previous_convergence = None
    stagnant_count = 0
    status_writer.write_event({
        "event": "run_started",
        "phase": "setup",
        "iteration": start_from_iteration,
        "base_save_path": base_save_path,
        "n_iterations": n_iterations,
        "robot_steps": robot_steps,
        "first_robot_steps": first_robot_steps,
        "hand_steps": hand_steps,
        "n_envs": n_envs,
        "enable_convergence_stop": enable_convergence_stop,
        "extractor_name": extractor_name,
        "aux_mode": aux_mode,
        "history_length": history_length,
        "history_mode": history_mode,
        "future_horizon": future_horizon,
        "include_opponent_id": include_opponent_id,
    })
    run_config_path = os.path.join(base_save_path, "run_config.json")
    with open(run_config_path, "w", encoding="utf-8") as f:
        json.dump({
            "extractor_name": extractor_name,
            "aux_mode": aux_mode,
            "history_length": history_length,
            "history_mode": history_mode,
            "future_horizon": future_horizon,
            "include_opponent_id": include_opponent_id,
            "scripted_hand_sample_prob": scripted_hand_sample_prob,
            "pfsp": {
                "window_size": 2000,
                "min_episodes": 20,
                "length_alpha": 1.0,
                "temperature": 1.0,
                "min_prob": 0.05,
            },
            "strategy_aux_weights": {
                "traj": strategy_traj_weight,
                "risk": strategy_risk_weight,
                "contrastive": strategy_contrastive_weight,
                "temperature": strategy_contrastive_temperature,
            },
        }, f, ensure_ascii=False, indent=2)

    # Get extractor class
    extractor_class = EXTRACTOR_MAP.get(extractor_name.lower())
    if extractor_class is None:
        print(f"Warning: Unknown extractor '{extractor_name}', using default (no custom extractor)")
        extractor_class = None
    else:
        print(f"Using feature extractor: {extractor_class.__name__}")

    if (history_length != 16 or history_mode != "motion") and extractor_class not in GRU_EXTRACTORS:
        raise ValueError("non-legacy history settings are supported only by --extractor gru")

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

        if include_opponent_id and current_robot_path is not None:
            print("[*] Robot obs dim changes with opponent id; reusing hand pool but starting a new Robot model.")
            current_robot_path = None

    print(f"➤ 初始对手池大小: {len(historical_hand_paths) + 1} (包含脚本手)")

    # 从 start_from_iteration 开始训练
    for iteration in range(start_from_iteration, n_iterations + 1):
        print(f"\n{'#'*60}")
        print(f"# Iteration {iteration}/{n_iterations}")
        print(f"{'#'*60}")

        iteration_path = os.path.join(base_save_path, f"iteration_{iteration}")
        os.makedirs(iteration_path, exist_ok=True)
        status_writer.write_event({
            "event": "iteration_started",
            "phase": "setup",
            "iteration": iteration,
            "opponent_pool_size": len(historical_hand_paths),
        })

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

        if include_opponent_id:
            resume_robot_path = None
            continue_from_previous = False

        print(f"➤ 当前对手池大小: {len(historical_hand_paths) + 1}")

        current_robot_opponent_id_dim = (1 + len(historical_hand_paths)) if include_opponent_id else 0

        # 1. 训练 Robot
        current_robot_steps = first_robot_steps if iteration == start_from_iteration else robot_steps
        current_robot_path = train_robot(
            hand_model_paths=historical_hand_paths,
            total_steps=current_robot_steps,
            save_path=os.path.join(iteration_path, "robot"),
            n_envs=n_envs,
            extractor_class=extractor_class,
            previous_robot_path=current_robot_path,
            continue_from_previous=continue_from_previous and (current_robot_path is not None),
            start_iteration=iteration,
            skip_if_exists=False,  # start_from 指定的轮次必须训练
            resume_from_path=resume_robot_path,  # 之前的轮次从这里加载
            status_run_dir=base_save_path,
            status_log_freq=status_log_freq,
            include_opponent_id=include_opponent_id,
            scripted_hand_sample_prob=scripted_hand_sample_prob,
            history_length=history_length,
            history_mode=history_mode,
            future_horizon=future_horizon,
            aux_mode=aux_mode,
            strategy_aux_weights={
                "traj": strategy_traj_weight,
                "risk": strategy_risk_weight,
                "contrastive": strategy_contrastive_weight,
                "temperature": strategy_contrastive_temperature,
            },
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
        prev_hand_warm_start = historical_hand_paths[-1] if historical_hand_paths else None

        new_hand_path = train_hand(
            robot_model_path=current_robot_path,
            total_steps=hand_steps,
            save_path=os.path.join(iteration_path, "hand"),
            n_envs=n_envs,
            extractor_class=extractor_class,
            skip_if_exists=False,  # start_from 指定的轮次必须训练
            resume_from_path=resume_hand_path,  # 之前的轮次（跳过训练时使用）
            warm_start_path=prev_hand_warm_start,  # 上一轮 Hand 模型（热启动）
            status_run_dir=base_save_path,
            status_log_freq=status_log_freq,
            iteration=iteration,
            robot_opponent_id_dim=current_robot_opponent_id_dim,
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
            status_writer.write_event({
                "event": "opponent_added",
                "phase": "league",
                "iteration": iteration,
                "hand_path": new_hand_path,
                "opponent_pool_size": len(historical_hand_paths),
            })

        if enable_convergence_stop and historical_hand_paths:
            print("[*] Running convergence evaluation...")
            current_convergence = evaluate_convergence(
                current_robot_path,
                historical_hand_paths,
                episodes=convergence_episodes,
                pool_score_lambda=pool_score_lambda,
            )
            decision = convergence_decision(
                current_convergence,
                previous_convergence,
                stagnant_count,
                patience=convergence_patience,
                z_value=convergence_z,
            )
            stagnant_count = decision["stagnant_count"]
            event = {
                "event": "convergence_evaluated",
                "phase": "convergence_eval",
                "iteration": iteration,
                "metrics": current_convergence,
                "decision": decision,
            }
            status_writer.write_event(event)
            append_json(convergence_history_path, event)
            previous_convergence = current_convergence

            print(
                f"[*] PoolScore={current_convergence['pool_score']:.4f}, "
                f"mean_TIS={current_convergence['mean_tis']:.4f}, "
                f"worst_TIS={current_convergence['worst_case_tis']:.4f}, "
                f"decision={decision['reason']}"
            )
            if decision["stop"]:
                print(f"[*] Convergence reached at iteration {iteration}. Stopping league training.")
                status_writer.write_event({
                    "event": "run_stopped_by_convergence",
                    "phase": "convergence_eval",
                    "iteration": iteration,
                    "decision": decision,
                })
                break

    print(f"\n{'='*60}")
    print("Iterative League Training Complete!")
    print(f"{'='*60}")
    print(f"Final Robot: {current_robot_path}")
    print(f"Total Opponents: {len(historical_hand_paths)}")
    status_writer.write_event({
        "event": "run_completed",
        "phase": "complete",
        "iteration": iteration if 'iteration' in locals() else start_from_iteration,
        "final_robot_path": current_robot_path,
        "total_opponents": len(historical_hand_paths),
    })

    return current_robot_path, historical_hand_paths[-1] if historical_hand_paths else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual agent iterative league training')
    parser.add_argument('--iterations', type=int, default=15, help='Number of training iterations')
    parser.add_argument('--robot_steps', type=int, default=2_000_000, help='Steps per iteration for robot after the first iteration')
    parser.add_argument('--first_robot_steps', type=int, default=3_000_000, help='Steps for first robot iteration')
    parser.add_argument('--hand_steps', type=int, default=1_000_000, help='Steps per iteration for hand')
    parser.add_argument('--n_envs', type=int, default=4, help='Number of parallel environments')
    parser.add_argument('--save_path', type=str, default=None, help='Save path')
    parser.add_argument('--extractor', type=str, default='mlp',
                        choices=['mlp', 'gru'],
                        help='Feature extractor to use')
    parser.add_argument('--aux', type=str, default='none',
                        choices=['none', 'single', 'multi_risk', 'contrastive'],
                        help='Auxiliary objective for GRU experiments')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to previous training run to continue from (e.g., logs/dual_iterative_0419_1041)')
    parser.add_argument('--start_from', type=int, default=1,
                        help='Start from this iteration number (e.g., 2 to continue from iteration 1)')
    parser.add_argument('--enable_convergence_stop', action='store_true',
                        help='Stop league training when pool score improvement is within statistical noise')
    parser.add_argument('--convergence_episodes', type=int, default=100,
                        help='Episodes per opponent for post-iteration convergence evaluation')
    parser.add_argument('--convergence_patience', type=int, default=2,
                        help='Consecutive stagnant iterations before stopping')
    parser.add_argument('--convergence_z', type=float, default=1.96,
                        help='Z value for statistical noise band')
    parser.add_argument('--pool_score_lambda', type=float, default=0.25,
                        help='Penalty weight for TIS variance across opponents')
    parser.add_argument('--status_log_freq', type=int, default=10000,
                        help='Timesteps between dashboard status writes')
    parser.add_argument('--include_opponent_id', action='store_true',
                        help='Append opponent one-hot id to robot observations')
    parser.add_argument('--scripted_hand_sample_prob', type=float, default=None,
                        help='Probability of sampling the scripted hand when agent hand pool is non-empty')
    parser.add_argument('--history_length', type=int, default=16,
                        help='History length in observations')
    parser.add_argument('--history_mode', type=str, default='motion', choices=['motion', 'interaction'],
                        help='History representation: motion deltas or interaction features')
    parser.add_argument('--future_horizon', type=int, default=8,
                        help='Future horizon for multi-step GRU auxiliary prediction')
    parser.add_argument('--strategy_traj_weight', type=float, default=0.1,
                        help='Weight for future trajectory auxiliary loss')
    parser.add_argument('--strategy_risk_weight', type=float, default=0.02,
                        help='Weight for future catch-risk auxiliary loss')
    parser.add_argument('--strategy_contrastive_weight', type=float, default=0.05,
                        help='Weight for supervised contrastive strategy-embedding auxiliary loss')
    parser.add_argument('--strategy_contrastive_temperature', type=float, default=0.1,
                        help='Temperature for supervised contrastive strategy-embedding auxiliary loss')

    args = parser.parse_args()

    run_iterative_training(
        n_iterations=args.iterations,
        robot_steps=args.robot_steps,
        first_robot_steps=args.first_robot_steps,
        hand_steps=args.hand_steps,
        n_envs=args.n_envs,
        base_save_path=args.save_path,
        extractor_name=args.extractor,
        resume_from=args.resume_from,
        start_from_iteration=args.start_from,
        enable_convergence_stop=args.enable_convergence_stop,
        convergence_episodes=args.convergence_episodes,
        convergence_patience=args.convergence_patience,
        convergence_z=args.convergence_z,
        pool_score_lambda=args.pool_score_lambda,
        status_log_freq=args.status_log_freq,
        include_opponent_id=args.include_opponent_id,
        scripted_hand_sample_prob=args.scripted_hand_sample_prob,
        aux_mode=args.aux,
        history_length=args.history_length,
        history_mode=args.history_mode,
        future_horizon=args.future_horizon,
        strategy_traj_weight=args.strategy_traj_weight,
        strategy_risk_weight=args.strategy_risk_weight,
        strategy_contrastive_weight=args.strategy_contrastive_weight,
        strategy_contrastive_temperature=args.strategy_contrastive_temperature,
    )
