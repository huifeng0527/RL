# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Paper Core

An end-to-end RL framework for adaptive upper-limb rehabilitation. A UR10 robot controls a magnetic microrobot on a desktop surface; the patient's hand tracks it. The robot learns to keep the interaction within the patient's Zone of Proximal Development (ZPD), providing personalized difficulty without hand-tuned controllers.

**Four contributions:**

1. **Cognitive-Motor Decoupled Domain Randomization (CMD-DR)**: Two-layer virtual patient generation. Layer 1: intent from scripted heuristics or trained RL opponents. Layer 2: biomechanical execution via low-pass filter (alpha=0.7), acceleration clipping (a_max=0.15), Gaussian observation noise, and neural delay (0-3 frames). Separates "what to do" from "how the body moves" to cover the full clinical spectrum.

2. **Dual-Stream Encoder + Auxiliary Forward Dynamics**: MLP processes spatial state (positions, boundaries); LSTM processes 16-frame relative displacement history (translation-invariant). A self-supervised head predicts next-frame hand displacement (MSE loss). Dense gradients from this auxiliary task compel the LSTM to internalize latent patient dynamics (tremor, delay, inertia).

3. **Iterative League Training with PFSP**: Robot and Hand trained alternately, not simultaneously. Hand agents accumulate in an opponent pool. Each new Hand is warm-started from the previous generation. Sampling uses Prioritized Fictitious Self-Play: inversely weighted by competitive balance, with a probability floor (0.05) to avoid neglecting any opponent.

4. **Sim-to-Real via Decoupled Vision-Control Architecture**: Async vision thread (camera-rate YOLO + MediaPipe) feeds a fixed-rate control thread (20Hz PPO inference). Dead-reckoning fills vision gaps using low-pass filtered velocity. Zero-shot transfer from simulation to UR10 physical platform.

## Codebase Layout

- **`src/`**: Simulation + training (gymnasium environments, PPO, league training)
- **`rlproject/`**: Real-world deployment (UR10, vision pipeline, FastAPI backend, React frontend)
- **`manuscripts/`**: IEEE paper drafts
- **`memory/`**: Project-specific experimental observations and analysis notes. Check this directory when reasoning about prior experiment results or league-training behavior.

## Commands

```bash
conda activate rl

# Dual-agent iterative league training
python src/scripts/train_dual_iterative.py --iterations 5 --robot_steps 1000000 --hand_steps 2000000
python src/scripts/train_dual_iterative.py --resume_from logs/dual_iterative_XXXX --start_from 2 --steps 3000000

# Ablation (5 feature extractor variants, 8M steps total)
python src/ab_study.py

# Testing
python src/scripts/test_dual.py --robot <robot.zip> --hand <hand.zip>
python src/scripts/test_mouse_hand.py --model <robot.zip>   # you play hand, agent plays robot
python src/scripts/test_mouse_robot.py --model <hand.zip>    # you play robot, agent plays hand

# Real-world
python rlproject/start_system.py              # full stack (backend:8000 + frontend:5173)
python rlproject/src/eval.py                  # 4 tasks: Sprint / Tracking / LeagueGame / Boundary
python rlproject/src/main.py                  # real-world fine-tuning
```

## Key Technical Details

- **Observation**: 44 dims = 12 scalar (positions, distance, boundaries, stride, prev action) + 32 temporal (16-frame displacement history)
- **Reward (Robot)**: ZPD band [4,6] gives +0.5; d<4 exponential penalty; d>=6 linear penalty; jerk penalty (-2); boundary violation (-80)
- **Feature extractors** (`src/utils/feature_extractors.py`): MLPOnly, LSTM, AuxLSTM, Gated, AuxGated
- **Aux training callback**: batch=512, aux_epochs=3, lr=5e-5, gradient clip 0.5, runs at PPO rollout end
- **Hand pool PFSP**: floor probability 0.05 per opponent; sampled by episode-length competitiveness
- **Real-world pipeline**: vision 10-15fps, control 20Hz, robot IP 192.168.1.2, camera 2592x1944
- **YOLO model**: `rlproject/src/runs/detect/train3/weights/best.onnx`
