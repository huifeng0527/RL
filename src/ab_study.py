"""Ablation study framework for rehabilitation RL.

Runs experiments with different feature extractor architectures.
"""

import os
import datetime
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from .custom_env import RehabilitationEnv
from .utils.feature_extractors import EXTRACTOR_REGISTRY
from .utils.ablation_callbacks import PPOAuxTrainingCallback, SaveMetricsCallback


def run_ablation_study(
    experiments=None,
    total_steps=5_000_000,
    n_envs=8,
    eval_freq=10000
):
    """Run ablation study with different feature extractors.

    Args:
        experiments: List of experiment configs. Each config is a dict with:
            - name: experiment name
            - extractor: feature extractor class
            - use_aux: whether to use auxiliary training
        total_steps: Total training steps per experiment
        n_envs: Number of parallel environments
        eval_freq: Evaluation frequency
    """
    if experiments is None:
        experiments = [
            {"name": "1_MLP_Only", "extractor": "MLPOnly", "use_aux": False},
            {"name": "2_MLP_LSTM", "extractor": "LSTM", "use_aux": False},
            {"name": "3_MLP_LSTM_AUX", "extractor": "AuxLSTM", "use_aux": True},
            {"name": "4_MLP_LSTM_FiLM", "extractor": "FiLM", "use_aux": False},
        ]

    now = datetime.datetime.now().strftime("%m%d_%H%M")
    base_log_dir = f"logs/ablation_study_{now}/"
    os.makedirs(base_log_dir, exist_ok=True)

    for exp in experiments:
        exp_name = exp["name"]
        extractor_name = exp["extractor"]
        use_aux = exp.get("use_aux", False)

        print(f"\n{'='*40}")
        print(f"Starting ablation: {exp_name}")
        print(f"{'='*40}\n")

        def make_env():
            env = RehabilitationEnv(training_mode='robot')
            return Monitor(env)

        vec_env = SubprocVecEnv([make_env for _ in range(n_envs)])

        # Resolve extractor class
        if isinstance(extractor_name, str):
            extractor_class = EXTRACTOR_REGISTRY[extractor_name]
        else:
            extractor_class = extractor_name

        policy_kwargs = dict(
            net_arch=[128, 256, 64],
            features_extractor_class=extractor_class,
            features_extractor_kwargs=dict(),
            share_features_extractor=True
        )

        save_path = os.path.join(base_log_dir, exp_name)
        tb_dir = os.path.join(base_log_dir, "tensorboard_logs")
        os.makedirs(save_path, exist_ok=True)

        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            learning_rate=3e-4,
            batch_size=512,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tb_dir
        )

        eval_cb = EvalCallback(
            vec_env,
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=eval_freq,
            n_eval_episodes=100,
            deterministic=True,
            render=False
        )
        metrics_cb = SaveMetricsCallback(save_path)

        if use_aux:
            aux_cb = PPOAuxTrainingCallback(batch_size=512)
            callbacks = CallbackList([eval_cb, aux_cb, metrics_cb])
        else:
            callbacks = CallbackList([eval_cb, metrics_cb])

        model.learn(
            total_timesteps=total_steps,
            callback=callbacks,
            tb_log_name=exp_name
        )

        model.save(os.path.join(save_path, "final_model.zip"))
        vec_env.close()
        del model

    print("\nAblation study complete! View with Tensorboard:")
    print(f"tensorboard --logdir {base_log_dir}")


if __name__ == '__main__':
    run_ablation_study()
