# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Architecture

This repository implements a rehabilitation training environment using reinforcement learning, where:
- `src/custom_env.py` contains `RehabilitationEnv`, a gymnasium environment for robot-hand interaction
- Training alternates between two modes (`training_mode='robot'` or `'hand'`)
- Hand behavior uses scripted movement (move toward or away from robot)
- Observations include position data, distance metrics, and 16-frame historical trajectories

Key architectural components:
- Opponent pool system in `RehabilitationEnv` for multi-agent training
- Biomechanical cost calculation for hand movement reward shaping
- Academic-style rendering with detailed hand visualization
- Dual perspective observations (`_get_robot_obs` and `_get_hand_obs`)

## Project Structure

```
src/
├── __init__.py          # Package exports
├── custom_env.py        # Core RehabilitationEnv class
├── renderer.py          # Academic-style rendering
├── training.py          # Training utilities
├── ab_study.py          # Ablation study framework
├── utils/
│   ├── __init__.py
│   └── callbacks.py     # DebugCallback
├── scripts/             # Training scripts
└── archive/             # Old code (main.ipynb)
```

## Development Commands

### Training
```bash
# Train robot agent
python -c "from src.training import train_robot; train_robot()"

# Train hand agent
python -c "from src.training import train_hand; train_hand('logs/robot_model/best_model.zip')"
```

### Testing
```bash
# Manual environment test
python -c "from src.custom_env import RehabilitationEnv; env = RehabilitationEnv(render_mode='human'); obs, _ = env.reset(); env.render()"
```

### Environment Setup
```bash
pip install gymnasium stable-baselines3 torch pygame numpy
```

## Key Files

- `src/custom_env.py`: Core environment implementation
  - `RehabilitationEnv` class with dual training modes
  - Scripted hand movement via `_get_scripted_hand_move()`
  - Biomechanical cost calculation for hand training
  - History buffers for trajectory tracking

- `src/renderer.py`: Academic-style visualization
  - Detailed hand drawing with fingers and nails
  - Trajectory rendering
  - Arm visualization

- `src/training.py`: Training utilities
  - `train_robot()`: Train robot to catch hand
  - `train_hand()`: Train hand to avoid robot

- `src/ab_study.py`: Ablation study framework with four feature extractors

## Special Notes

- Observation structure: 10 scalar dimensions + 32 historical dimensions (16 frames × 2 channels)
- Hand opponent pool uses `None` for scripted behavior, or loads trained models
- APF module has been removed - robot uses direct RL action
- Do not modify the observation dimension calculations without updating all feature extractors