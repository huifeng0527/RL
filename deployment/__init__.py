"""
部署代码模块
用于实际机器人控制
"""
import os
import sys

# 确保可以导入common模块
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

