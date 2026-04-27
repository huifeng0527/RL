"""Train Baseline B: Robot trained only with single strongest RL hand.

This creates the contrast to PFSP - instead of a pool of diverse hands,
Baseline B only ever sees one hand (iteration_9's best hand).
Uses the same training setup as train_dual_iterative.py for consistency.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.scripts.train_dual_iterative import train_robot as train_robot_with_extractor
from src.utils.feature_extractors import AuxLSTMExtractor

# Path to the single strongest hand (iteration_9)
STRONGEST_HAND = r"C:\Users\admin\Desktop\科研\RL\logs\dual_iterative_0427_1314\iteration_9\hand\hand\best_model.zip"

# Save path for Baseline B - the script will create iteration_1/robot/robot subdirs
SAVE_PATH = r"C:\Users\admin\Desktop\科研\RL\logs\dual_iterative_0427_1314\baseline_b"

if __name__ == '__main__':
    print("="*60)
    print("Training Baseline B: Single RL Hand opponent")
    print("="*60)
    print(f"Hand opponent: iteration_9 hand (strongest)")
    print(f"Save path: {SAVE_PATH}")
    print(f"Feature extractor: AuxLSTMExtractor")
    print("="*60)

    train_robot_with_extractor(
        hand_model_paths=[STRONGEST_HAND],  # Single hand only - no pool
        total_steps=3_000_000,
        save_path=SAVE_PATH,
        n_envs=8,
        extractor_class=AuxLSTMExtractor,
        skip_if_exists=False,
    )

    print("\n" + "="*60)
    print("Baseline B training complete!")
    print(f"Model should be at: {SAVE_PATH}/iteration_1/robot/robot/best_model.zip")
    print("="*60)
