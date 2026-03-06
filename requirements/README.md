# 依赖安装说明

本项目将依赖分为三个部分：

## 1. 通用依赖 (common.txt)
基础科学计算和工具库，所有环境都需要。

## 2. 仿真依赖 (simulation.txt)
用于训练强化学习模型的环境，包括：
- Gymnasium: 强化学习环境框架
- Stable-Baselines3: 强化学习算法库
- PyTorch: 深度学习框架
- Pygame: 环境可视化

## 3. 部署依赖 (deployment.txt)
用于实际机器人控制的环境，包括：
- Stable-Baselines3: 加载训练好的模型
- RTDE: UR机器人控制接口
- OpenCV: 图像处理
- MediaPipe: 手部检测
- Ultralytics: YOLO目标检测

## 安装方法

### 仅安装仿真环境
```bash
pip install -r requirements/simulation.txt
```

### 仅安装部署环境
```bash
pip install -r requirements/deployment.txt
```

### 安装所有依赖
```bash
pip install -r requirements/simulation.txt
pip install -r requirements/deployment.txt
```

## 注意事项

1. PyTorch需要根据您的CUDA版本单独安装，请访问 https://pytorch.org/ 获取正确的安装命令
2. RTDE库需要UR机器人的RTDE接口支持
3. 某些依赖可能需要系统级别的库（如OpenCV需要ffmpeg等）

