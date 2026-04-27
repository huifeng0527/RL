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
└── rlproject/                    # Real-world deployment (前后端分离系统)
    ├── backend/                  # FastAPI 后端 (port 8000)
    │   ├── main.py               # REST API + WebSocket
    │   ├── database.py           # SQLite 数据库模型
    │   ├── eval_engine.py        # 评估引擎 (4种任务)
    │   └── report_generator.py   # PDF 报告生成
    ├── frontend/                # React + Vite 前端 (port 5173)
    │   └── src/
    │       ├── pages/            # Patients, PatientDetail, Evaluation, History
    │       ├── components/       # Layout 等组件
    │       └── services/api.js   # API 调用封装
    └── start_system.py           # 前后端统一启动脚本
```

## 前后端系统

### 启动方式
```bash
python rlproject/start_system.py              # 启动前后端
python rlproject/start_system.py --backend-only   # 只启动后端
python rlproject/start_system.py --frontend-only  # 只启动前端
```

### API 端点
- `GET/POST /api/patients` - 患者管理
- `GET/POST /api/sessions` - 评估会话
- `PATCH /api/sessions/{id}/notes` - 更新备注
- `DELETE /api/sessions/{id}` - 删除会话（同时删除录像）
- `POST /api/eval/start` - 开始评估
- `WebSocket /ws/eval` - 实时进度、帧、FPS 推送

### 数据库
- SQLite: `rlproject/backend/rehab_eval.db`
- 表: patients, sessions, eval_sprint, eval_tracking, eval_league, eval_boundary
- Session 表新增 `video_path` 字段存储评估录像路径

### 评估任务 (EvalEngine)
1. **Sprint** - 反应与爆发力（5次目标捕捉）
2. **Tracking** - 多轨迹追踪（Circle + Figure-8）
3. **LeagueGame** - 对抗与安全距离（RL机器人追击）
4. **Boundary** - 活动范围与稳定性（矩形边界追踪）

### 前端组件
- `Evaluation.jsx` - 实时显示 FPS、进度、任务状态
- `History.jsx` - 评估历史，支持备注编辑和删除
- 录像保存于 `rlproject/videos/eval_session_{id}_{timestamp}.mp4`

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
- Feature extractors in `utils/feature_extractors.py`: MLPOnlyExtractor, LSTMExtractor, AuxLSTMExtractor, GatedExtractor, AuxGatedExtractor

## Hardware Configuration

| Component | Configuration |
|-----------|--------------|
| Robot IP | 192.168.1.2 |
| Control Frequency | 25Hz (dt=0.04s) |
| Camera Resolution | 2592x1944 |
| YOLO Model | `RL/rlproject/src/runs/detect/train3/weights/best.onnx` |

## Dependencies

```bash
pip install gymnasium stable-baselines3 torch pygame numpy ultralytics mediapipe opencv-python
```
## Rule
在对话结束，请给我推送一句名言，中文也好，英文也好