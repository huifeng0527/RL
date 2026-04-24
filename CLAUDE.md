# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a rehabilitation training system using reinforcement learning. It has two main components:

1. **Simulation** (`src/`): Train RL agents in simulation
2. **Deployment** (`rlproject/`): Deploy trained models on real hardware (UR robot)

## Architecture

```
RL/
├── src/                          # Simulation training
│   ├── custom_env.py             # RehabilitationEnv (gymnasium environment)
│   ├── renderer.py               # Academic-style visualization
│   ├── training.py               # train_robot() / train_hand() utilities
│   ├── ab_study.py               # Ablation study with 4 feature extractors
│   ├── scripts/
│   │   └── train_dual_iterative.py  # Dual agent iterative training
│   └── utils/
│       ├── callbacks.py          # DebugCallback
│       ├── feature_extractors.py # MLP, LSTM, Aux extractors
│       └── ablation_callbacks.py # PPOAuxTrainingCallback
│
└── rlproject/                    # Real-world deployment
    └── src/
        ├── main.py               # Real-world fine-tuning with PPO
        ├── eval.py               # Evaluation tasks (Sprint/Tracking/LeagueGame/Boundary)
        └── custom_env/           # Deployment-specific environment
```

## Core Training Logic

**Dual Agent Training** (`train_dual_iterative.py`):
- `training_mode='robot'`: Robot learns to catch hand (uses `hand_model_pool` for opponents)
- `training_mode='hand'`: Hand learns to avoid robot (uses scripted movement)


**Observation Structure**:
- `10 + 16*2 = 42` dimensions: position(2) + velocity(2) + distance(1) + bounds(4) + stride(1) + history(32)

## Development Commands

**IMPORTANT: Always use conda environment `rl` first**
```bash
conda activate rl
```

### Simulation Training
```bash
# Start new training
python src/scripts/train_dual_iterative.py --iterations 5 --steps 5000000

# Resume from previous training (--start_from N means iteration N must train)
python src/scripts/train_dual_iterative.py --resume_from logs/dual_iterative_0422_1812 --start_from 2 --steps 3000000
```

**Resume behavior**:
- Models from iterations < `start_from` are loaded from `resume_from` and copied to new directory
- Iterations >= `start_from` always train (models not skipped)
- Previous iteration's robot is used as starting point when continuing training

### Testing
```bash
# Test dual agents
python src/scripts/test_dual.py --robot <robot_model.zip> --hand <hand_model.zip>

# Mouse control (you control robot, agent plays hand)
python src/scripts/test_mouse_robot.py --model <hand_model.zip>

# Mouse control (you control hand, agent plays robot)
python src/scripts/test_mouse_hand.py --model <robot_model.zip>

# Cross evaluation heatmap
python src/scripts/cross_eval.py --base_dir logs/dual_iterative_0419_1041 --iterations 5
```

### Deployment (rlproject)
```bash
cd rlproject/src
python main.py              # Real-world fine-tuning
python eval.py              # Evaluation tasks
```

## Key Implementation Notes

- When `training_mode='hand'`, no hand model sampling occurs - scripted movement is used
- `hand_model_pool` is only populated when `training_mode='robot'` and `hand_model_paths` is provided
- Observation dimension: 10 scalar + 32 history (16 frames x 2 channels) = 42 total
- APF module removed - robot uses direct RL action output
- Feature extractors in `utils/feature_extractors.py`: MLPOnlyExtractor, LSTMExtractor, AuxLSTMExtractor, GatedExtractor, AuxGatedExtractor

## Hardware Configuration

| Component | Configuration |
|-----------|--------------|
| Robot IP | 192.168.1.2 |
| Control Frequency | 25Hz (dt=0.04s) |
| Camera Resolution | 2592x1944 |
| YOLO Model | `rlproject/src/runs/detect/train3/weights/best.onnx` |

## Dependencies

```bash
pip install gymnasium stable-baselines3 torch pygame numpy ultralytics mediapipe opencv-python
```
