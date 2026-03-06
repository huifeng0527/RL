# 机器人康复训练系统

基于强化学习的机器人康复训练系统，包含仿真训练和实际部署两个部分。

## 项目结构

```
RL/
├── simulation/              # 仿真代码目录
│   ├── environments/       # 环境定义
│   ├── training/           # 训练脚本
│   ├── configs/            # 配置文件
│   ├── utils/              # 工具函数
│   └── notebooks/          # Jupyter notebooks
│
├── deployment/              # 部署代码目录
│   ├── main.py             # 部署主程序
│   ├── configs/            # 部署配置
│   ├── robot_control/       # 机器人控制
│   ├── vision/             # 视觉处理
│   ├── models/             # 模型文件
│   └── utils/               # 部署工具
│
├── common/                 # 共享代码
│   └── observation_space.py
│
└── requirements/           # 依赖管理
    ├── common.txt
    ├── simulation.txt
    └── deployment.txt
```

## 快速开始

### 1. 安装依赖

#### 仿真环境
```bash
pip install -r requirements/simulation.txt
```

#### 部署环境
```bash
pip install -r requirements/deployment.txt
```

### 2. 配置

#### 仿真配置
编辑 `simulation/configs/env_config.yaml` 和 `simulation/configs/train_config.yaml`

#### 部署配置
编辑 `deployment/configs/` 目录下的配置文件：
- `robot_config.yaml`: 机器人IP和运动参数
- `camera_config.yaml`: 摄像头和标定文件路径
- `model_config.yaml`: 模型路径和控制参数
- `hand_detection_config.yaml`: 手部检测参数

### 3. 训练模型（仿真）

```python
from simulation.environments import CustomEnv
from stable_baselines3 import SAC
import yaml

# 加载环境配置
with open('simulation/configs/env_config.yaml', 'r') as f:
    env_config = yaml.safe_load(f)

# 创建环境
env = CustomEnv(config=env_config)

# 训练模型
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("deployment/models/best_model")
```

### 4. 部署运行

```bash
cd deployment
python main.py
```

## 配置说明

### 环境配置 (simulation/configs/env_config.yaml)

主要参数：
- `grid_size`: 网格大小
- `distance_threshold_*`: 各种距离阈值
- `reward_*`: 奖励参数
- `stride_*_range`: 步长范围

### 训练配置 (simulation/configs/train_config.yaml)

主要参数：
- `algorithm`: 使用的算法（SAC/PPO等）
- `total_timesteps`: 总训练步数
- `learning_rate`: 学习率
- `eval_freq`: 评估频率

### 部署配置

#### robot_config.yaml
- `robot_ip`: UR机器人IP地址
- `robot_speed`: 运动速度
- `robot_orientation`: 机器人姿态

#### camera_config.yaml
- `camera_id`: 摄像头ID
- `desired_width/height`: 图像分辨率
- `calibration_matrix_path`: 相机标定文件路径
- `homography_matrix_path`: 单应性矩阵路径

#### model_config.yaml
- `model_path`: 训练好的模型路径
- `yolo_model_path`: YOLO检测模型路径
- `env_width/height`: 环境尺寸
- `control_frequency`: 控制频率（Hz）

## 使用说明

### 仿真训练

1. 配置环境参数和训练参数
2. 运行训练脚本或Jupyter notebook
3. 模型会自动保存到指定路径

### 实际部署

1. 确保机器人、摄像头连接正常
2. 配置所有部署配置文件
3. 将训练好的模型放到 `deployment/models/` 目录
4. 运行 `python deployment/main.py`
5. 按 'q' 键退出

## 注意事项

1. **路径配置**: 所有配置文件中的路径都是相对于 `deployment/` 目录的
2. **模型兼容性**: 确保部署时使用的模型与训练时的环境配置一致
3. **硬件要求**: 
   - 仿真环境：GPU推荐用于加速训练
   - 部署环境：需要UR机器人RTDE接口和摄像头
4. **安全**: 实际部署时请确保机器人安全区域设置正确

## 故障排除

### 常见问题

1. **模型加载失败**: 检查模型路径和观察空间/动作空间是否匹配
2. **摄像头无法打开**: 检查摄像头ID和权限
3. **机器人连接失败**: 检查IP地址和网络连接
4. **坐标转换错误**: 检查相机标定文件和单应性矩阵是否正确

## 开发说明

### 添加新功能

1. **新环境**: 在 `simulation/environments/` 中添加
2. **新算法**: 在训练脚本中使用stable-baselines3的其他算法
3. **新传感器**: 在 `deployment/vision/` 中添加处理模块

### 代码规范

- 使用类型提示
- 添加文档字符串
- 遵循PEP 8代码风格

## 许可证

[添加您的许可证信息]

## 联系方式

[添加联系方式]

