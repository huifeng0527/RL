"""Train Baseline B: Robot trained only with a single RL hand.

This creates the contrast to PFSP: instead of a pool of diverse hands,
Baseline B only ever sees one selected hand opponent.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.scripts.train_dual_iterative import train_robot as train_robot_with_extractor
from src.utils.feature_extractors import AuxLSTMExtractor


def default_hand_path(base_dir, hand_generation):
    return os.path.join(
        base_dir,
        f"iteration_{hand_generation}",
        "hand",
        "hand",
        "best_model.zip",
    )


def main():
    parser = argparse.ArgumentParser(description="Train Baseline B: single RL hand opponent")
    parser.add_argument("--base_dir", default=None)
    parser.add_argument("--hand_generation", type=int, default=9)
    parser.add_argument("--hand_path", default=None)
    parser.add_argument("--robot_path", default=None)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--n_envs", type=int, default=8)
    args = parser.parse_args()

    strongest_hand = args.hand_path or default_hand_path(args.base_dir, args.hand_generation)
    save_path = args.save_path or os.path.join(args.base_dir, "baselines", "single_rl_hand")

    if not os.path.exists(strongest_hand):
        raise FileNotFoundError(f"Hand model not found: {strongest_hand}")

    print("=" * 60)
    print("Training Baseline B: Single RL Hand opponent")
    print("=" * 60)
    print(f"Hand opponent: {strongest_hand}")
    print(f"Robot init: {args.robot_path or 'from scratch'}")
    print(f"Save path: {save_path}")
    print("Feature extractor: AuxLSTMExtractor")
    print("=" * 60)

    train_robot_with_extractor(
        hand_model_paths=[strongest_hand],
        total_steps=args.steps,
        save_path=save_path,
        n_envs=args.n_envs,
        extractor_class=AuxLSTMExtractor,
        skip_if_exists=False,
        resume_from_path=args.robot_path,
    )

    print("\n" + "=" * 60)
    print("Baseline B training complete!")
    print(f"Model should be at: {save_path}/robot/best_model.zip")
    print("=" * 60)


if __name__ == '__main__':
    main()
