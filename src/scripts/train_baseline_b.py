"""Train Baseline B: robot trained against one frozen learned hand only."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.scripts.train_dual_iterative import train_robot
from src.utils.feature_extractors import StrategyGRUAuxExtractor


DEFAULT_BASE_DIR = r"C:\Users\admin\Desktop\research\RL\logs\league_paper_gru_multistep_aux_pfsp_window_20iter"


def default_hand_path(base_dir, hand_generation):
    return os.path.join(
        base_dir,
        f"iteration_{hand_generation}",
        "hand",
        "hand",
        "best_model.zip",
    )


def main():
    parser = argparse.ArgumentParser(description="Train Baseline B: single learned-hand opponent")
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--hand_generation", type=int, default=10)
    parser.add_argument("--hand_path", default=None)
    parser.add_argument("--robot_path", default=None)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--history_length", type=int, default=16)
    parser.add_argument("--history_mode", choices=["motion", "interaction"], default="interaction")
    parser.add_argument("--aux_mode", choices=["none", "single", "multi_risk", "contrastive"], default="none")
    parser.add_argument("--future_horizon", type=int, default=8)
    args = parser.parse_args()

    hand_path = args.hand_path or default_hand_path(args.base_dir, args.hand_generation)
    save_path = args.save_path or os.path.join(args.base_dir, "baselines", f"single_hand_h{args.hand_generation}")

    if not os.path.exists(hand_path):
        raise FileNotFoundError(f"Hand model not found: {hand_path}")

    print("=" * 60)
    print("Training Baseline B: Single Learned Hand Opponent")
    print("=" * 60)
    print(f"Hand opponent: {hand_path}")
    print(f"Robot init: {args.robot_path or 'from scratch'}")
    print(f"Save path: {save_path}")
    print("Feature extractor: StrategyGRUAuxExtractor")
    print(f"History: {args.history_mode}, length={args.history_length}")
    print(f"Auxiliary mode: {args.aux_mode}, future_horizon={args.future_horizon}")
    print("=" * 60)

    train_robot(
        hand_model_paths=[hand_path],
        total_steps=args.steps,
        save_path=save_path,
        n_envs=args.n_envs,
        extractor_class=StrategyGRUAuxExtractor,
        skip_if_exists=False,
        resume_from_path=args.robot_path,
        scripted_hand_sample_prob=0.0,
        history_length=args.history_length,
        history_mode=args.history_mode,
        future_horizon=args.future_horizon,
        aux_mode=args.aux_mode,
    )

    print("\n" + "=" * 60)
    print("Baseline B training complete!")
    print(f"Model should be at: {save_path}/robot/best_model.zip")
    print("=" * 60)


if __name__ == "__main__":
    main()
