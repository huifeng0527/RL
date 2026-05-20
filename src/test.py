import os
import sys

# 添加项目根目录到 sys.path，使得模型加载时可以找到 src 模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stable_baselines3 import PPO

model = PPO.load(os.path.join(project_root, "src", "logs", "dual_iterative_0509_0945", "iteration_10", "robot", "robot", "best_model.zip"))