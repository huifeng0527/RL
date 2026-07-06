"""Train Baseline A: robot trained against the scripted hand only."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.scripts.train_dual_iterative import train_robot
from src.utils.feature_extractors import StrategyGRUAuxExtractor


DEFAULT_BASE_DIR = "logs/league_paper_gru_multistep_aux_pfsp_window_20iter"


def main():
    parser = argparse.ArgumentParser(description="Train Baseline A: scripted hand only")
    parser.add_argument("--base_dir", default=DEFAULT_BASE_DIR)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--steps", type=int, default=3_000_000)
    parser.add_argument("--n_envs", type=int, default=8)
    parser.add_argument("--history_length", type=int, default=16)
    parser.add_argument("--history_mode", choices=["motion", "interaction"], default="interaction")
    parser.add_argument("--aux_mode", choices=["none", "single", "multi_risk", "contrastive"], default="none")
    parser.add_argument("--future_horizon", type=int, default=8)
    args = parser.parse_args()

    save_path = args.save_path or os.path.join(args.base_dir, "baselines", "scripted_only")

    print("=" * 60)
    print("Training Baseline A: Scripted Hand Only")
    print("=" * 60)
    print(f"Save path: {save_path}")
    print("Feature extractor: StrategyGRUAuxExtractor")
    print(f"History: {args.history_mode}, length={args.history_length}")
    print(f"Auxiliary mode: {args.aux_mode}, future_horizon={args.future_horizon}")
    print("=" * 60)

    train_robot(
        hand_model_paths=[],
        total_steps=args.steps,
        save_path=save_path,
        n_envs=args.n_envs,
        extractor_class=StrategyGRUAuxExtractor,
        skip_if_exists=False,
        scripted_hand_sample_prob=1.0,
        history_length=args.history_length,
        history_mode=args.history_mode,
        future_horizon=args.future_horizon,
        aux_mode=args.aux_mode,
    )

    print("\n" + "=" * 60)
    print("Baseline A training complete!")
    print(f"Model should be at: {save_path}/robot/best_model.zip")
    print("=" * 60)


if __name__ == "__main__":
    main()
