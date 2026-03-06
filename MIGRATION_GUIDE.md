# 迁移指南

本文档说明如何从旧代码结构迁移到新的重构后的结构。

## 目录结构变化

### 旧结构
```
RL/
├── src/                    # 仿真代码（混合）
├── rlproject/src/          # 部署代码（混合）
│   ├── custom_env/         # 环境定义
│   ├── cv/                 # 视觉处理
│   ├── robot_control/      # 机器人控制
│   └── main.py             # 部署主程序
```

### 新结构
```
RL/
├── simulation/             # 仿真代码（清晰分离）
│   ├── environments/       # 环境定义
│   ├── training/           # 训练脚本
│   ├── configs/            # 配置文件
│   └── utils/              # 工具函数
├── deployment/             # 部署代码（清晰分离）
│   ├── configs/            # 部署配置
│   ├── robot_control/      # 机器人控制
│   ├── vision/             # 视觉处理
│   └── main.py             # 部署主程序
├── common/                 # 共享代码
└── requirements/           # 依赖管理
```

## 代码迁移

### 1. 环境代码迁移

**旧代码位置：**
- `src/custom_env.py`
- `rlproject/src/custom_env/env.py`

**新代码位置：**
- `simulation/environments/custom_env.py`（统一实现）

**使用方式：**
```python
# 旧方式
from custom_env import CustomEnv

# 新方式
from simulation.environments import CustomEnv
```

### 2. 部署代码迁移

**旧代码位置：**
- `rlproject/src/main.py`
- `rlproject/src/cv/*`
- `rlproject/src/robot_control/*`

**新代码位置：**
- `deployment/main.py`
- `deployment/vision/*`
- `deployment/robot_control/*`

**使用方式：**
```python
# 旧方式
from cv.hand_detect import HandDetection
from robot_control.ur_control import URControl

# 新方式
from deployment.vision.hand_detection import HandDetection
from deployment.robot_control.ur_control import URControl
```

### 3. 配置迁移

**旧方式：** 硬编码在代码中

**新方式：** 使用YAML配置文件

**示例：**
```python
# 旧方式（硬编码）
robot_ip = "192.168.1.2"
env_width = 15
env_height = 10

# 新方式（配置文件）
from deployment.utils.config_loader import ConfigLoader
config_loader = ConfigLoader()
robot_config = config_loader.get_robot_config()
robot_ip = robot_config['robot_ip']
```

## 配置文件设置

### 1. 环境配置

编辑 `simulation/configs/env_config.yaml`：
```yaml
grid_size: 10
distance_threshold_penalty: 5
# ... 其他参数
```

### 2. 训练配置

编辑 `simulation/configs/train_config.yaml`：
```yaml
algorithm: "SAC"
total_timesteps: 100000
learning_rate: 0.0003
# ... 其他参数
```

### 3. 部署配置

编辑 `deployment/configs/*.yaml`：
- `robot_config.yaml`: 机器人IP和运动参数
- `camera_config.yaml`: 摄像头和标定文件路径
- `model_config.yaml`: 模型路径和控制参数
- `hand_detection_config.yaml`: 手部检测参数

## 路径更新

### 模型路径

**旧方式：**
```python
model = SAC.load(r"C:\Users\admin\Desktop\huifeng\RL\rlproject\src\model1\model_500step.zip")
```

**新方式：**
```yaml
# deployment/configs/model_config.yaml
model_path: "./models/best_model.zip"
```

```python
from deployment.utils.config_loader import ConfigLoader
config_loader = ConfigLoader()
model_config = config_loader.get_model_config()
model_path = config_loader.resolve_path(model_config['model_path'])
model = SAC.load(model_path)
```

### 标定文件路径

**旧方式：**
```python
cali = CameraCalibration(
    calibration_matrix_path='./camera_calibration/calibration_data.npz',
    homography_matrix_path='./camera_calibration/Homography_matrix.npy'
)
```

**新方式：**
```yaml
# deployment/configs/camera_config.yaml
calibration_matrix_path: "./vision/calibration_data.npz"
homography_matrix_path: "./vision/Homography_matrix.npy"
```

## 运行方式变化

### 训练

**旧方式：**
```bash
# 在Jupyter notebook中运行
```

**新方式：**
```bash
# 方式1：使用训练脚本
python simulation/training/train.py

# 方式2：在代码中使用
from simulation.training.train import train_model
train_model()
```

### 部署

**旧方式：**
```bash
cd rlproject/src
python main.py
```

**新方式：**
```bash
cd deployment
python main.py
```

## 注意事项

1. **模型兼容性**：确保新代码加载的模型与训练时的环境配置一致
2. **路径配置**：所有配置文件中的路径都是相对于各自目录的
3. **依赖安装**：需要重新安装依赖（使用新的requirements文件）
4. **数据迁移**：需要将模型文件、标定文件等复制到新目录结构

## 向后兼容

为了平滑过渡，可以：
1. 保留旧目录一段时间
2. 在新代码中添加兼容性导入
3. 逐步迁移配置文件

## 常见问题

### Q: 如何迁移现有的模型文件？
A: 将模型文件复制到 `deployment/models/` 目录，并更新 `model_config.yaml` 中的路径。

### Q: 标定文件在哪里？
A: 将标定文件复制到 `deployment/vision/` 目录，并更新 `camera_config.yaml` 中的路径。

### Q: 如何运行旧的notebook？
A: 可以更新notebook中的导入路径，或者使用新的训练脚本。

## 获取帮助

如果遇到问题，请检查：
1. 配置文件路径是否正确
2. 依赖是否已正确安装
3. 模型文件是否存在且路径正确
4. 查看README.md获取更多信息

