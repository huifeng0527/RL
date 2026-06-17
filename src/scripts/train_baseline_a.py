"""Train Baseline A: Robot trained against scripted hand only."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.scripts.train_dual_iterative import train_robot
from src.utils.feature_extractors import AuxLSTMExtractor


def main():
    parser = argparse.ArgumentParser(description="Train Baseline A: scripted hand only")
    parser.add_argument("--base_dir", default="src/logs/dual_iterative_0509_0945")
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--n_envs", type=int, default=8)
    args = parser.parse_args()

    save_path = args.save_path or os.path.join(args.base_dir, "baselines", "scripted_only")

    print("=" * 60)
    print("Training Baseline A: Scripted Hand Only")
    print("=" * 60)
    print(f"Save path: {save_path}")
    print("Feature extractor: AuxLSTMExtractor")
    print("=" * 60)

    train_robot(
        hand_model_paths=[],
        total_steps=args.steps,
        save_path=save_path,
        n_envs=args.n_envs,
        extractor_class=AuxLSTMExtractor,
        skip_if_exists=False,
    )

    print("\n" + "=" * 60)
    print("Baseline A training complete!")
    print(f"Model should be at: {save_path}/robot/best_model.zip")
    print("=" * 60)


if __name__ == "__main__":
    main()
